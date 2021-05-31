"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains routines to process microscopy data and images.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import tifffile

from lklib.util import isnone, printmsg
from lklib.fileread import read_bin_file, change_extension
from lklib.cfgparse import read_cfg
from lklib.image import crop_rem_rrcc, corr_field_illum, get_frac_sat_rng, \
    remap_img, bake_cmap, gen_preview_img

from lcmicro.common import DataType, MosaicType
from lcmicro.cfgparse import get_idx_mask, get_scan_field_size, \
    get_scan_frame_time, get_scan_px_sz, get_px_time, get_ex_rep_rate, \
    get_cfg_range, get_cfg_gamma, get_data_type, get_nl_ord, \
    get_def_chan_idx, get_px_cnt_limit, get_px_bckgr_count, get_idx_ts_ms, \
    get_idx_z_pos, get_chan_name, get_chan_filter_name, get_microscope_name, \
    parse_chan_idx

from lcmicro.polarimetry.dataio import load_nsmp


def get_chan_frames(data=None, config=None, chan=2):
    """Get frames from the specified channel."""
    return data[:, :, get_idx_mask(config, chan)]


def get_chan_sum(**kwargs):
    """Sum the counts in the frames of the given channels."""
    return get_chan_frames(**kwargs).sum(2)


def get_img(**kwargs):
    """Just get an image."""
    [img, _, _, _] = proc_img(**kwargs)
    return img


def export_chans(config=None, data=None, ch_id=None, rng=None):
    """Export channels."""
    if isnone(config):
        raise Exception("NoConfig")

    if isnone(ch_id):
        ch_id = get_def_chan_idx(config)

    idx_arr = get_idx_mask(config, ch_id)
    sat_thr = get_px_cnt_limit(config)
    bgr_lvl = get_px_bckgr_count(config)

    if isnone(rng):
        rng = [0, sat_thr]

    for ind in enumerate(idx_arr):
        data = data[:, :, idx_arr[ind]]
        num_sig = np.sum(data > bgr_lvl)
        num_sat = np.sum(data > sat_thr)
        num_over = np.sum(data > rng[1])
        img_file = r".\img\img_{:d}_{:d}.png".format(ch_id, ind)

        print(
            "Saving file " + img_file
            + ". Signal: {:d} px, sat: {:d} px, over: {:d} px".format(
                num_sig, num_sat, num_over))

        plt.imsave(img_file, data, vmin=rng[0], vmax=rng[1], cmap="gray")


def get_def_chan_cmap(config, ch=2):
    """Get default colormap based on channel name."""
    ch_name = get_chan_name(config, chan=ch)
    if ch_name == "DAPI":
        return "KPW_Nice"
    elif ch_name == "SHG":
        return "viridis"
    elif ch_name == "THG":
        return "inferno"
    else:
        return get_def_cmap(get_chan_filter_name(config, chan=ch))


def get_def_cmap(chan_str=None):
    """Get the default colourmap based on the channel description."""
    if chan_str is None:
        return "magma"
    if chan_str.find("BP340") != -1:
        return "KBW_Nice"
    if chan_str.find("BP520") != -1 or chan_str.find("BP550") != -1:
        return "KGW_Nice"
    if chan_str.find("BP600") != -1:
        return "KOW_Nice"
    if chan_str.find("BP650") != -1:
        return "KRW_Nice"
    return "magma"


def estimate_spot_sz(wavl=1.028, NA=0.75, n=1):
    """Estimate the lateral and axial spot size."""
    w0_xy = 0.318*wavl/NA
    wh_xy = 1.18*w0_xy

    wh_z = 0.61*wavl/(n-np.sqrt(n**2 - NA**2))

    print("NA = {:.2f}, lamda = {:.3f} um, n = {:.3f}".format(
        NA, wavl, n))
    print("XY = {:.2f} um FWHM, Z = {:.2f} um FWHM".format(
        wh_xy, wh_z))


def print_cnt_lin_info(cnt, dwell_t=None, frep=None):
    """Print counting linearity and correction info."""
    if isnone(dwell_t):
        print("Assuming 1 s dwell time")
        dwell_t = 1
    if isnone(frep):
        print("Assuming 75 MHz repetition rate")
        frep = 75e6

    rate = cnt/dwell_t
    print("Count rate: {:.3f} Mcps".format(rate/1e6))

    prob = rate/frep
    print("Count probability: {:.3f}".format(prob))

    frep = 0.5*prob**2
    print("Correction factor: {:3g}".format(frep))

    fsev = frep/prob
    print("Correction severity: {:3g}".format(fsev))

    cnt_corr = cnt*(1+fsev)
    print("Corrected counts: {:.3f} Mcps".format(cnt_corr/1e6))

    print("Count delta: {:.3f} Mcps".format((cnt_corr - cnt)/1e6))
    print("Correction severity: {:.3f}".format((cnt_corr - cnt)/cnt))


def get_scan_artefact_sz(file_name=None, config=None, **kwargs):
    """Get the size of the flyback scan artefact.

    Get the size of the scan artefact on the left side of the image due to the
    galvo mirror flyback. The size is in pixels.

    Scan artefact size depends on the scan amplitude (i.e. the scan field size)
    and the scan speed (i.e. the frame time) in a nontrivial manner. The
    dependency on scan speed seems to have a sigmoidal relationship. As the
    speed decreases the artefact becomes smaller, but only up to a certain
    extent set by the scan amplitude and mirror inertia. As the speed increases
    the artefact grows but up to a certain extent where the scan amplitude
    becomes almost sinusoidal and the artefact is half the image. As a result
    the artefact size is quite difficult to estimate in general so here an
    empirical approach is taken.
    """
    verbosity = kwargs.get('verbosity')

    if isnone(config):
        config = read_cfg(file_name)

    field_sz_um = get_scan_field_size(config)
    frame_t_s = get_scan_frame_time(config)
    umpx = get_scan_px_sz(config)

    if field_sz_um is None or frame_t_s is None or umpx is None:
        print("Cannot determine scan artefact size")
        return None

    scope_name = get_microscope_name(config)

    if scope_name == 'LCM1':
        # Observed artefact sizes for given scan field sizes and frame times
        field_sz = [780, 393, 157, 78, 39]
        artefact_sz_arr = (
            (  # 780 µm
            [10],  # Frame time in s
            [41.5]),  # Artefact size in µm
            (  # 393 µm
            [2.5],
            [27.5]),
            (  # 157 µm
            [0.8],
            [16.5]),
            (  # 78 µm
            [1],
            [3.14]),
            (  # 39 µm
            [1],
            [2.4]))
    elif scope_name == 'FF':
        # Derived from LCM1 calibration data based on 2020.10.13 reference
        # measurements. The data point at 450 µm field size, 10 s ir actual
        # reference measurements, the rest are linearly adjusted from LCM1
        # data.
        # sz,   ft,     px,     asz
        # 500,  22.5,   0.33,   39.9
        # 500,  10,     ,       57.4

        # 450,  22.5,   0.3,    35.4
        field_sz = [500, 450, 400]
        artefact_sz_arr = [
            (  # 500 µm
            [22.5, 10],
            [39.9, 57.5]),
            (  # 450 µm
            [22.5, 10],
            [35.4, 47.3]),
            (  # 400 µm
            [40],
            [21])]
    else:
        print("Unknown microscope. Cannot retrieve scan artifact size.")
        return None

    # The scan flyback artefact depends on several factors, mostly on the scan
    # field size, then on the frame scan time.

    # Find the closest calibration scan field size
    ind_sz = field_sz.index(min(field_sz, key=lambda x: abs(x-field_sz_um)))

    # Find the closest calibration frame time
    frame_t_arr = artefact_sz_arr[ind_sz][0]
    ind_ft = frame_t_arr.index(
        min(frame_t_arr, key=lambda x: abs(x-frame_t_s)))

    # Assume linear scaling with deviation from the calibration scan time to
    # the corresponding scan field size.
    t_fac = frame_t_arr[ind_ft]/frame_t_s

    crop_sz_arr = artefact_sz_arr[ind_sz][1]
    crop_sz = crop_sz_arr[ind_ft]

    # umpx seems to play a role as well. For a field size of 780 and pixel size
    # of 0.78 um the artefact is 42 um, but when pixel size is 0.39 um the
    # artefact becomes 91 um for some reason.
    # if(ind == 0 and umpx < 0.31):
    #     crop_sz = 91
    # else:
    #     crop_sz = crop_sz_arr[ind]

    # Apply frame time scaling
    crop_sz = crop_sz*t_fac

    # Convert crop size in um to px
    crop_sz_px = int(crop_sz/umpx)

    if verbosity == 'info':
        print("Scan artefact size is {:.1f} µm, {:d} px".format(
            crop_sz, crop_sz_px))

    return crop_sz_px


def crop_scan_artefacts(img, config, **kwargs):
    """Crop away galvo-scanning image artefacts."""
    crop_sz_px = get_scan_artefact_sz(config=config, **kwargs)
    img = crop_rem_rrcc(img, 0, 0, crop_sz_px, 0)
    return img


def get_sat_mask(img, config):
    """Get a mask showing saturated pixels in an image."""
    px_t = get_px_time(config)
    frep = get_ex_rep_rate(config)

    if px_t is None or frep is None:
        print("Cannot determine saturation level")
        return None

    sat_level = frep/10 * px_t
    mask = img / sat_level
    return mask


def proc_img(
        file_name=None, rng=None, gamma=None, ch=2, corr_fi=False,
        crop_artefacts=True, **kwargs):
    """Process an image for analysis and display.

    Obtain specified mapping range and gamma values, crop scan artefacts and
    correct field illumination.
    """
    verbosity = kwargs.get('verbosity')
    data = read_bin_file(file_name)
    config = read_cfg(file_name)

    if rng is not None:
        if verbosity == 'info':
            print("Using supplied mapping range: [{:d}, {:d}]".format(
                rng[0], rng[1]))
    else:
        rng = get_cfg_range(config, chan_id=ch)
        if rng is not None and verbosity == 'info':
            print("Using config mapping range: [{:d}, {:d}]".format(
                rng[0], rng[1]))

    if gamma is None:
        gamma = get_cfg_gamma(config, ch=ch)

    if gamma is None:
        gamma = 1

    data_type = get_data_type(config=config)
    if data_type in [
            DataType.SingleImage, DataType.Average, DataType.TimeLapse]:
        if data_type == DataType.SingleImage:
            img = data[:, :, ch]

        if data_type in [DataType.Average, DataType.TimeLapse]:
            img = get_chan_sum(data=data, config=config, chan=ch)

        # Convert image to volts for analog channels
        # Assuming channel range is +-10V, no offset and 16bits
        if ch in (0, 1):
            img = (img.astype('float')/2**16 - 0.5)*20

        if crop_artefacts:
            if verbosity == 'info':
                print("Cropping scan artefacts...")
            img = crop_scan_artefacts(img, config, **kwargs)
        else:
            if verbosity == 'info':
                print("Scan artefact cropping disabled")

        if corr_fi:
            if verbosity == 'info':
                print("Correcting field illumination...")
            img = corr_field_illum(img, facpwr=get_nl_ord(config, ch))
        else:
            if verbosity == 'info':
                print("Field illumination correction disabled")

        if isnone(rng):
            rng = get_opt_map_rng(img=img, file_name=file_name, **kwargs)

    return [img, rng, gamma, data]


def load_pipo(file_name=None, chan_ind=None, binsz=None,
              cropsz=None, pad_to_128=False):
    """Load dataset as a PIPO map.

    TODO: This function skips many of the data parsing routines implemented
    in e.g. report image generation. These should be integrated here.

    If binsz == 'all', the images in the dataset are summed to a single pixel.
    """
    config = read_cfg(file_name)
    chan_ind = parse_chan_idx(config, chan_ind)

    print("Channel index: {:d}".format(chan_ind))

    print("Reading '{:s}'...".format(file_name), end='', flush=True)
    data = read_bin_file(file_name)
    print('OK', flush=True)

    num_chan = 4
    num_img = int(data.shape[2]/num_chan)
    if num_img == 65:
        data = data[:, :, :-4]
        num_img = int(data.shape[2]/num_chan)
        print("There are 65 images in the dataset, assuming this is 8x8 PIPO "
              "with an extra garbage state.")

    num_psg_states = num_psa_states = np.sqrt(num_img)
    if num_psg_states - int(num_psg_states) != 0:
        print("There are {:d} images in the dataset, ".format(num_img) +
              "which does not correspond to any NxN PIPO sequence")

    num_psg_states = int(num_psg_states)
    num_psa_states = int(num_psa_states)
    if binsz == 'all':
        pipo_iarr = np.ndarray([num_psa_states, num_psg_states])
    else:
        num_row, num_col = np.shape(data)[0:2]
        if cropsz:
            num_row = cropsz[1] - cropsz[0]
            num_col = cropsz[3] - cropsz[2]
        pipo_iarr = np.ndarray(
            [num_row, num_col, num_psa_states, num_psg_states])

    if cropsz:
        print("Cropping image to " + str(cropsz) + " px")

    print("Assuming PSA-first order")
    for ind_psg in range(num_psg_states):
        for ind_psa in range(num_psa_states):
            frame_ind = (ind_psa + ind_psg*num_psa_states)*num_chan + chan_ind
            img = data[:, :, frame_ind]
            if cropsz:
                img = img[cropsz[0]:cropsz[1], cropsz[2]:cropsz[3]]
            if binsz == 'all':
                pipo_iarr[ind_psa, ind_psg] = np.sum(img)
            else:
                pipo_iarr[:, :, ind_psa, ind_psg] = img
    if pad_to_128:
        trg_sz = [128, 128]
        [num_row, num_col] = pipo_iarr.shape[0:2]
        if num_row < trg_sz[0] or num_col < trg_sz[1]:
            print("Padding image to {:d}x{:d} px".format(trg_sz[0], trg_sz[1]))

        pipo_iarr2 = np.ndarray([trg_sz[0], trg_sz[1], num_psa_states, num_psg_states])
        pipo_iarr2.fill(0)

        row_from = np.max([int((trg_sz[0] - num_row)/2), 0])
        row_to = row_from + num_row
        col_from = np.max([int((trg_sz[1] - num_row)/2), 0])
        col_to = col_from + num_col

        pipo_iarr2[row_from:row_to, col_from:col_to, :, :] = pipo_iarr
        pipo_iarr = pipo_iarr2

    return pipo_iarr


def convert_pipo_to_tiff_piponator(**kwargs):
    """Convert a PIPO dataset to a PIPONATOR TIFF."""
    return convert_pipo_to_tiff(**kwargs, preset='piponator')


def convert_pipo_to_tiff(
        file_name=None, pipo_arr=None, preset='basic',
        out_sz=None,
        duplicate_first_and_last_state=False, add_dummy_ref_states=False,
        reverse_psg_axis=False, sanity_check=False,
        **kwargs):
    """Convert a PIPO dataset to 16-bit multipage TIFF.

    TODO: This function is very similar to convert_nsmp_to_tiff, the two should
    probably be merged.

    Multipage TIFF files are useful for moving data to other software. The
    16-bit output ensures raw count numbers are maintained. Note, that not all
    software supports 16-bit TIFF files.

    The dataset can be supplied either directly (pipo_arr) or by specifying a
    file name (file_name) to load the dataset from.

    The 2D PIPO polarization states of the input dataset will be arranged
    linearly as pages in the output TIFF file in PSA-major order with
    reference states, if present, interleaved after each PSA cycle.

    Output TIFF formatting can be adjusted either using presets or by
    configuring format parameters directly.

    Output image size can be specified using out_sz. If the output size is
    different from the input image size, the images are resampled.

    If the input dataset does not contain reference states, and
    add_dummy_ref_states is True, dummy reference states are created by
    duplicating the first input (PSG=0, PSA=0) state. This is useful when
    the output must contain the reference states to maintain a particular
    state order.

    If the input dataset has unique PSG and PSA angle positions, and
    duplicate_first_and_last_state to True, the last state is produced by
    duplicating the first one. This is useful when the output must contain
    samples at 0 and 180 PSG and PSA degrees, even though they should be the
    same.

    The reverse_psg_axis flag can be set to reverse the PSG angle order. This
    is useful, when there is an axis reversal in the source data or for buggy
    data importers.

    For the TIFF file to be supported by PIPONATOR, the image size has to be
    128x128, and the PIPO array must contain interleaved reference
    states (which are called bleach states in PIPONATOR). Furthermore, the
    first and last PSG and PSA states must be identical, i.e. be at 0 deg and
    180 deg. The interleaving sequence is:
        PSG=0, PSA=0;
        PSG=0, PSA=1;
        ...
        PSG=0, PSA=N-1;
        REF State;
        PSG=1, PSA=0;
        PSG=1, PSA=1;
        ...
        PSG=N-1; PSA=N-1;
        REF State

    In addition, the reverse_psg_axis flag has to be set. It seems PIPONATOR
    has a PSG axis reverse bug at least in v12.38.
    """
    if preset not in ['basic', 'piponator']:
        print("Unsupported preset")
        return None

    swap_psa_psg = False
    if preset == 'piponator':
        out_sz = 128
        duplicate_first_and_last_state = True
        add_dummy_ref_states = True
        reverse_psg_axis = True
        swap_psa_psg = False
    elif preset == 'basic':
        pass

    if pipo_arr is None:
        pipo_arr = load_pipo(file_name, **kwargs).astype('uint16')

    num_row, num_col, num_psg, num_psa = np.shape(pipo_arr)
    print("Input dataset size: {:d}x{:d} px, {:d}x{:d} PSGxPSA".format(
        num_row, num_col, num_psg, num_psa))

    if num_row != num_col:
        print("This function was only tested for square images")

    num_total_counts = np.sum(pipo_arr)

    print("Total number of counts: {:d}".format(int(num_total_counts)))
    if num_total_counts == 0:
        print("WARNING: Channel has zero counts, the PMT was likely off of the "
              "signal cable was not connected.")

    if duplicate_first_and_last_state:
        print("Output dataset with duplicate first and last PSG and PSA states"
              " requested, but the input dataset contains unique PSG and PSA "
              "states. PSG=0 and PSA=0 states will be duplicated and "
              "appended.")
        # Add +1 to number of PSG and PSA states that will be filled using the
        # the available input states
        num_psg += 1
        num_psa += 1

    num_states = num_psg*num_psa

    if add_dummy_ref_states:
        print("Output dataset with reference states requested, but the input "
              "dataset does not contain any. The PSG=0, PSA=0 state will be "
              "duplicated to make a dummy reference set.")
        # Reference states are usually measured after each PSA cycle. There are
        # as many reference states as there are PSG states, for each of which a
        # PSA state cycle is performed. Add this number to the total number of
        # states
        num_states += num_psg

    # Build output TIFF file name
    tiff_file_name = change_extension(file_name, '.tiff')

    # Initialize output
    if out_sz is None:
        out_row = num_row
        out_col = num_col
    else:
        out_row = out_col = out_sz

    pipo_arr_out = np.ndarray([num_states, out_row, out_col])

    if num_row != out_row or num_col != out_col:
        print("Image will be resampled to 128x128")

    # Calculate the factor to resample the input image to output size
    # Even though this takes the minimum over x/y, the function will likely
    # not work for non-square images.
    resample_fac = np.min([out_row/num_row, out_col/num_col])

    # Build PSG index range based on reverse flag
    if reverse_psg_axis:
        psg_range = range(num_psg-1, -1, -1)
    else:
        psg_range = range(num_psg)

    ind_state = 0
    img_ref = None

    # Loop over PSG and PSA to build the linear array for TIFF pages
    for ind_psg in psg_range:
        for ind_psa in range(num_psa):
            # ind_psg, ind_psa loop over logical states, which take data
            # reodering and duplication flags into account
            # ind_psg_in, ind_psa_in index the actual input data
            #
            # Start by assuming the logical and actual indices are the same
            ind_psg_in = ind_psg
            ind_psa_in = ind_psa

            if duplicate_first_and_last_state:
                # When duplicating take the 0-th actual index when the logical
                # index is last
                if ind_psg == num_psg - 1:
                    ind_psg_in = 0

                if ind_psa == num_psa - 1:
                    ind_psa_in = 0

            # Take the input image and resample it to output size
            # Even though a 'zoom' function sounds funny, it works on 16-bit
            # data stored as ndarray. PIL, OpenCV, etc. either need conversion
            # back and forth or don't even work on uint16 data
            if swap_psa_psg:
                img_out = ndimg.zoom(
                    pipo_arr[:, :, ind_psg_in, ind_psa_in], resample_fac)
            else:
                img_out = ndimg.zoom(
                    pipo_arr[:, :, ind_psa_in, ind_psg_in], resample_fac)

            if add_dummy_ref_states and img_ref is None:
                # Copy the PSG=0, PSA=0 state to use as a dummy for all
                # reference states
                img_ref = img_out.copy()

            # Assign the output image in a linearly incrementing way
            # The linear increment is crucial as it allows easy reference state
            # interleaving
            pipo_arr_out[ind_state, :, :] = img_out
            ind_state += 1

            if add_dummy_ref_states and ind_psa == num_psa - 1:
                # Add a reference state after each PSA cycle
                pipo_arr_out[ind_state, :, :] = img_ref
                ind_state += 1

    print("Output dataset size: {:d}x{:d}p px, {:d} interleaved states".format(
        out_row, out_col, np.shape(pipo_arr_out)[0]))

    if sanity_check:
        # Perform a sanity check by undoing all data formatting steps and
        # checking whether the output data is the same as the input
        check = pipo_arr[0, 0, :, :]
        data = pipo_arr_out[:, 0, 0]

        if add_dummy_ref_states:
            data = np.delete(data, np.arange(-1, num_states, num_psg+1)[1:])

        data = np.reshape(data, [num_psa, num_psg])

        if reverse_psg_axis:
            data = np.flipud(data)

        if duplicate_first_and_last_state:
            data = data[:-1, :-1]

        if not np.all(data == check):
            plt.subplot(1, 3, 1)
            plt.imshow(data)
            plt.title('Output data')
            plt.subplot(1, 3, 2)
            plt.imshow(check)
            plt.title('Check')
            plt.subplot(1, 3, 3)
            plt.imshow(data-check, cmap='coolwarm')
            plt.title('Error')
            plt.show()
            raise Exception("TIFF export sanity check failed")

    print("Writing '{:s}'".format(tiff_file_name))
    tifffile.imwrite(tiff_file_name, pipo_arr_out)
    print("All done")


def convert_nsmp_to_tiff(
        file_name=None, pimg_arr=None, preset='basic',
        out_sz=None, add_dummy_ref_states=False, sanity_check=False,
        **kwargs):
    """Convert an NSMP dataset to a 16-bit multipage TIFF.

    Multipage TIFF files are useful for moving data to other software. The
    16-bit output ensures raw count numbers are maintained. Note, that not all
    software supports 16-bit TIFF files.

    The dataset can be supplied either directly (pimg_arr) or by specifying a
    file name (file_name) to load the dataset from.

    The input pimg_arr is a 4D array of polarization-resolved images in a 2D
    PSG-PSA grid. The polarization grid is arranged linearly as pages in the
    output TIFF file in PSA-major order with reference states, if present,
    interleaved after each PSA cycle.

    Output TIFF formatting can be adjusted either using presets or by
    configuring format parameters directly.

    Output image size can be specified using out_sz. If the output size is
    different from the input image size, the images are resampled.

    If the input dataset do not contain reference states, and
    add_dummy_ref_states is True, dummy reference states are created by
    duplicating the first input (0, 0 index) state. This is useful when
    the output TIFF must contain the reference states to maintain a particular
    state order.
    """
    if pimg_arr is None:
        pimg_arr = load_nsmp(file_name, **kwargs).astype('uint16')

    num_row, num_col, num_psa, num_psg = np.shape(pimg_arr)
    print("Input dataset size: {:d}x{:d}p px, {:d}x{:d} PSGxPSA".format(
        num_row, num_col, num_psg, num_psa))

    if num_row != num_col:
        print("This function was only tested for square images")

    num_states = num_psg*num_psa

    if add_dummy_ref_states:
        print("Output dataset with reference states requested, but the input "
              "dataset does not contain any. The PSG=0, PSA=0 state will be "
              "duplicated to make a dummy reference set.")
        # Reference states are usually measured after each PSA cycle. There are
        # as many reference states as there are PSG states, for each of which a
        # PSA state cycle is performed. Add this number to the total number of
        # states
        num_states += num_psg

    # Build output TIFF file name
    tiff_file_name = change_extension(file_name, '.tiff')

    # Initialize output
    if out_sz is None:
        out_row = num_row
        out_col = num_col
    else:
        out_row = out_col = out_sz

    pimg_arr_out = np.ndarray([num_states, out_row, out_col])

    if num_row != out_row or num_col != out_col:
        print("Image will be resampled to 128x128")

    # Calculate the factor to resample the input image to output size
    # Even though this takes the minimum over x/y, the function will likely
    # not work for non-square images.
    resample_fac = np.min([out_row/num_row, out_col/num_col])

    ind_state = 0
    img_ref = None

    # Loop over PSG and PSA to build the linear array for TIFF pages
    for ind_psg in range(num_psg):
        for ind_psa in range(num_psa):
            # Take the input image and resample it to output size
            # Even though a 'zoom' function sounds funny, it works on 16-bit
            # data stored as ndarray. PIL, OpenCV, etc. either need conversion
            # back and forth or don't even work on uint16 data
            img_out = ndimg.zoom(
                pimg_arr[:, :, ind_psa, ind_psg], resample_fac)

            if add_dummy_ref_states and img_ref is None:
                # Copy the PSG=0, PSA=0 state to use as a dummy for all
                # reference states
                img_ref = img_out.copy()

            # Assign output images in a linearly incrementing way
            # The linear increment is crucial as it allows easy reference state
            # interleaving
            pimg_arr_out[ind_state, :, :] = img_out
            ind_state += 1

            if add_dummy_ref_states and ind_psa == num_psa - 1:
                # Add a reference state after each PSA cycle
                pimg_arr_out[ind_state, :, :] = img_ref
                ind_state += 1

    print("Output dataset size: {:d}x{:d}p px, {:d} interleaved states".format(
        out_row, out_col, np.shape(pimg_arr_out)[0]))

    if sanity_check:
        # Perform a sanity check by undoing all data formatting steps and
        # checking whether the output data is the same as the input
        check = pimg_arr[0, 0, :, :]
        data = pimg_arr_out[:, 0, 0]

        if add_dummy_ref_states:
            data = np.delete(data, np.arange(-1, num_states, num_psg+1)[1:])

        data = np.reshape(data, [num_psa, num_psg])

        if not np.all(data == check):
            plt.subplot(1, 3, 1)
            plt.imshow(data)
            plt.title('Output data')
            plt.subplot(1, 3, 2)
            plt.imshow(check)
            plt.title('Check')
            plt.subplot(1, 3, 3)
            plt.imshow(data-check, cmap='coolwarm')
            plt.title('Error')
            plt.show()
            raise Exception("TIFF export sanity check failed")

    print("Writing '{:s}'".format(tiff_file_name))
    tifffile.imwrite(tiff_file_name, pimg_arr_out)
    print("All done")


def make_mosaic_img(data=None, mask=None, ij=None, pad=0.02, remap=True,
                    rng=None):
    """Arrange images into a mosaic with given coordinates and padding."""
    if isnone(rng):
        rng = [0, 20]
    [nr, nc, _] = data.shape
    pad_px = np.int32(np.round(max([nr, nc])*pad))

    num_grid_rows = ij[:, 0].max() + 1
    num_grid_cols = ij[:, 1].max() + 1

    mosaic_r = num_grid_rows*nr + (num_grid_rows-1)*pad_px
    mosaic_c = num_grid_cols*nc + (num_grid_cols-1)*pad_px

    if remap:
        mos_dtype = np.float64
    else:
        mos_dtype = data.dtype

    mos = np.ndarray([mosaic_r, mosaic_c], dtype=mos_dtype)
    for ind, indd in enumerate(mask):
        [grid_row, grid_col] = ij[ind, :]

        row_ofs = grid_row*(nr + pad_px)
        col_ofs = grid_col*(nc + pad_px)

        if remap:
            img = remap_img(data[:, :, indd], rng=rng)[0]
        else:
            img = data[:, :, indd]

        mos[row_ofs: nr + row_ofs, col_ofs: nc + col_ofs] = np.fliplr(img)

    return mos


def get_opt_map_rng(img=None, file_name=None, **kwargs):
    """Get optimal mapping range for an image."""
    vlvl = kwargs.get('verbosity')
    if isnone(file_name):
        print("Dataset file name has to be provided")
        return None

    printmsg("Estimating optimal data mapping range to 1% saturation.",
             'info', vlvl)

    dtype = get_data_type(file_name=file_name)

    if dtype == DataType.Tiling:
        # Using the tiling module results in a circular import. There are
        # several options available to avoid that:
        #   1) a data container class would hide the implementation of the
        #       range estimation, but this requires a big code overhaul
        #   2) range estimation could be a part of a different module that
        #       imports both tiling and proc
        #   3) basic mosaicing shouldn't require tiling functionality as the
        #       images simply have to be placed side by side. make_mosaic()
        #       should do that for multichannel data
        printmsg("Range estimation for tiled images doesn't yet work",
                 'warning', vlvl)
        return None

        # TODO: This should be done by make_mosaic_img
        # print("Crating dummy mosaic...")
        # if isnone(data) or isnone(mask) or isnone(ij):
        #     [data, mask, ij] = get_tiling_data(
        #         data=data, file_name=file_name)
        # img = make_mosaic_img(data, mask, ij, remap=False)

    printmsg("Determining optimal mapping range...", 'info', vlvl)
    rng = get_frac_sat_rng(img)

    printmsg("Mapping range: [{:d} , {:d}]".format(rng[0], rng[1]),
             'info', vlvl)
    return rng


def make_image(
        img=None, data=None, file_name=None, rng=None, gamma=None, cmap=None,
        cmap_sat=False, map_scale='lin', **kwargs):
    """Make an image for display."""
    vlvl = kwargs.get('verbosity')
    if isnone(img):
        [img, rng, gamma, data] = proc_img(
            file_name=file_name, rng=rng, gamma=gamma, **kwargs)

    config = read_cfg(file_name)

    data_type = get_data_type(config=config)
    if data_type in (DataType.SingleImage, DataType.Average,
                     DataType.TimeLapse):
        img_raw = img

        if cmap_sat:
            map_cap = False
        else:
            map_cap = True

        [img, rng] = remap_img(img, rng=rng, gamma=gamma, cap=map_cap)
        img_scaled = img

        if isnone(cmap):
            cmap = get_def_chan_cmap(config)

        if map_scale == 'log':
            printmsg('Using logarithmic scaling', 'info', vlvl)
            unique_lvl = np.unique(img)
            if unique_lvl[0] == 0:
                step = np.diff(unique_lvl)[0]
                if np.std(np.diff(unique_lvl)) < 1E-10:
                    img_log = np.log10(img+step)
                    img_log -= np.min(img_log)
                    img_log = img_log/np.max(img_log)*255
                    img = img_log
                else:
                    printmsg('Cannot apply logarithmic scaling', 'error', vlvl)

        img = bake_cmap(img/255, cmap=cmap, remap=False, cm_over_val='r',
                        cm_under_val='b')
    else:
        if data_type == DataType.TimeLapse:
            mos_type = MosaicType.TimeSeries
        elif data_type == DataType.ZStack:
            mos_type = MosaicType.ZStack
        else:
            print("Unknown data type" + str(data_type))

        if isnone(data):
            data = read_bin_file(file_name)

        show_mosaic(data, file_name, mos_type=mos_type)

    return [img, img_raw, img_scaled, cmap, rng, gamma]


def show_mosaic_img(**kwargs):
    """Make and show a channel mosaic image."""
    mosaic = make_mosaic_img(**kwargs)
    plt.imshow(mosaic)
    plt.axis('off')
    return mosaic


def make_mosaic(data, file_name, aspect=16/9, index_mask=None, det_ch=2):
    """Make a mosaic of images in a 3D array."""
    [nr, nc, nd] = data.shape
    pad = np.int32(np.round(max([nr, nc])*0.1))

    if index_mask is None:
        config = read_cfg(file_name)
        mask = get_idx_mask(config, det_ch)

    nd = mask.size

    n_mc = np.int32(np.ceil(np.sqrt(aspect*nd)))
    n_mr = np.int32(np.ceil(nd/n_mc))

    mosaic = np.ndarray([nr*n_mr + (n_mr-1)*pad, nc*n_mc + (n_mc-1)*pad, 4],
                        dtype='uint8')
    mosaic.fill(255)

    image_coords = np.ndarray([nd, 2])

    ind_mr = 0
    ind_mc = 0
    for ind_mos in range(nd):
        indd = mask[ind_mos]
        img = gen_preview_img(data[:, :, indd])

        from_r = ind_mr*(nr + pad)
        to_r = from_r + nr
        from_c = ind_mc*(nc + pad)
        to_c = from_c + nc
        mosaic[from_r: to_r, from_c: to_c, :] = img*255

        image_coords[ind_mos, :] = [from_r, from_c]

        ind_mc = ind_mc + 1
        if ind_mc == n_mc:
            ind_mr += 1
            ind_mc = 0

    return [mosaic, image_coords]


def show_mosaic(data, file_name, mos_type=None, aspect=16/9, index_mask=None,
                det_ch=2):
    """Show a mosaic of images in a 3D array."""
    config = read_cfg(file_name)
    if index_mask is None:
        mask = get_idx_mask(config, det_ch)

    [mos, image_coords] = make_mosaic(data, file_name, mos_type, aspect)

    nc = data.shape[1]

    plt.imshow(mos)
    plt.axis('off')

    if mos_type is None:
        mos_type = MosaicType.TimeSeries

    if mos_type == MosaicType.TimeSeries:
        lbl = get_idx_ts_ms(config, mask)/1000
        label_str_pre = 't= '
        label_str_suf = ' s'
    elif mos_type == MosaicType.ZStack:
        lbl = get_idx_z_pos(config, mask)
        label_str_pre = 'z= '
        label_str_suf = ' mm'
    else:
        print('Unknown mosaic type ' + str(mos_type))

    for ind in range(image_coords.shape[0]):  # pylint: disable=E1136
        cap_str = str(lbl[ind])
        if ind == 0:
            cap_str = label_str_pre + cap_str + label_str_suf
        plt.text(
            image_coords[ind, 1] + nc/2, image_coords[ind, 0] - 7, cap_str,
            horizontalalignment='center')


def make_composite_img(file_names, method="CombineToRGB", ofs=None, chas=None,
                       corr_fi=True):
    """Make a composite RGB image."""
    if method == "CombineToRGB":
        nch = len(file_names)
        for ind in range(0, nch):

            data = make_image(file_names[ind], ch=2, corr_fi=corr_fi)

            if ind == 0:
                [nr, nc] = data[2].shape
                img = np.ndarray([nr, nc, 3])  # RGB output image
                img_raw = np.ndarray([nr, nc, nch])
                img_scaled = np.ndarray([nr, nc, nch])
                cmap = []
                rng = []
                gamma = []

            if not isnone(chas):
                ch_ind = chas[ind]
            else:
                ch_ind = ind

            img_raw[:, :, ind] = data[1]
            img_scaled[:, :, ind] = data[2]/255
            cmap.append(data[3])
            rng.append(data[4])
            gamma.append(data[5])

            ofs_xy = ofs[ind]

            if ofs_xy is not None:
                ofs_x = ofs[ind][0]
                ofs_y = ofs[ind][1]
                img[:-ofs_y, :-ofs_x, ch_ind] = img_scaled[ofs_y:, ofs_x:, ind]
            else:
                img[:, :, ch_ind] = img_scaled[:, :, ind]
        return [img, img_raw, img_scaled, cmap, rng, gamma]
    elif method == "BlendRGB":
        print("RGB blending is not yet working")

        # TODO: Make RGB blending work
        # I0 = (D0[0])[:,:,0:3]
        # a0 = D0[1].astype(float)
        # I1 = (D1[0])[:,:,0:3]
        # a1 = D1[1].astype(float)

        # a0 = a0/a0.max()
        # a1 = a1/a1.max()

        # a0 = a0**0.5

        # a0 = a0/(a0+a1)
        # a1 = a1/(a0+a1)

        # I = .alpha_blend(I0, I1, a1=a0, a2=a1)

        # scipy.misc.imsave('I0.png', I0)
        # scipy.misc.imsave('I1.png', I1)
        # scipy.misc.imsave('I.png', I)

        return None
    else:
        print("Unknown method" + method)

    return None


def export_zstack_images(file_name, rng=None, chan_id=3):
    """Read Z-stack bin file and export images to files."""
    print("Reading file Z-stack file {:s}...".format(file_name))
    D = read_bin_file(file_name)
    print("Done")

    config = read_cfg(file_name)
    chan_id = 3

    mask = get_idx_mask(config, chan_id)
    print("Found {:d} images with channel id {:d}".format(len(mask), chan_id))

    if isnone(rng):
        print("Mapping range not specified, setting to [0, max_value/10]")
        vmin = 0
        vmax = 1
        for ind in range(len(mask)):
            max_val = np.max(D[:, :, mask[ind]])
            if(max_val > vmax):
                vmax = max_val

        rng = [int(vmin), int(vmax/10)]

    print("Using [{:d}, {:d}] mapping range".format(rng[0], rng[1]))
    print("Saving images...")
    for ind in range(len(mask)):
        img_file = r".\img\img_{:d}.png".format(ind)
        print("Saving file " + img_file)
        plt.imsave(img_file, D[:, :, mask[ind]], vmin=rng[0], vmax=rng[1],
                   cmap="gray")

    print("All done")
