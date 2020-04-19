"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains routines to process microscopy data and images.

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

from lklib.util import isnone
from lklib.fileread import read_bin_file
from lklib.cfgparse import read_cfg
from lklib.image import crop_rem_rrcc, corr_field_illum, get_frac_sat_rng, \
    remap_img, bake_cmap, gen_preview_img

from lcmicro.common import DataType, MosaicType
from lcmicro.cfgparse import get_idx_mask, get_scan_field_size, \
    get_scan_frame_time, get_scan_px_sz, get_px_time, get_ex_rep_rate, \
    get_cfg_range, get_cfg_gamma, get_data_type, get_nl_ord, \
    get_def_chan_idx, get_px_cnt_limit, get_px_bckgr_count, get_idx_ts_ms, \
    get_idx_z_pos, get_chan_name, get_chan_filter_name

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

    print("NA = %.2f, lamda = %.3f um, n = %.3f" %(NA, wavl, n))
    print("XY = %.2f um FWHM, Z = %.2f um FWHM" %(wh_xy, wh_z))

def print_cnt_lin_info(cnt, dwell_t=None, frep=None):
    """Print counting linearity and correction info."""
    if isnone(dwell_t):
        print("Assuming 1 s dwell time")
        dwell_t = 1
    if isnone(frep):
        print("Assuming 75 MHz repetition rate")
        frep = 75e6

    rate = cnt/dwell_t
    print("Count rate: %.3f Mcps" %(rate/1e6))

    prob = rate/frep
    print("Count probability: %.3f" %prob)

    frep = 0.5*prob**2
    print("Correction factor: %3g" %frep)

    fsev = frep/prob
    print("Correction severity: %3g" %fsev)

    cnt_corr = cnt*(1+fsev)
    print("Corrected counts: %.3f Mcps" %(cnt_corr/1e6))

    print("Count delta: %.3f Mcps" %((cnt_corr - cnt)/1e6))
    print("Correction severity: %.3f" %((cnt_corr - cnt)/cnt))

def get_scan_artefact_sz(file_name=None, config=None):
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
    if isnone(config):
        config = read_cfg(file_name)

    field_sz_um = get_scan_field_size(config)
    frame_t_s = get_scan_frame_time(config)
    umpx = get_scan_px_sz(config)

    if field_sz_um is None or frame_t_s is None or umpx is None:
        print("Cannot determine scan artefact size")
        return None

    # Observed artefact sizes for given scan field sizes and frame times
    field_sz = [780, 393, 157, 78, 39] # in um
    frame_t = [10, 2.5, 0.8, 1, 1] # in s
    crop_sz_arr = [41.5, 27.5, 16.5, 3.14, 2.4] # in um

    # Scan field size seems to be the largest factor. Find the closest
    # empirical scan field size for the current one
    ind = field_sz.index(min(field_sz, key=lambda x: abs(x-field_sz_um)))

    # Assume linear scaling with deviation from the empirical scan time for
    # the corresponding scan field size.
    t_fac = frame_t[ind]/frame_t_s

    crop_sz = crop_sz_arr[ind]

    # umpx seems to play a role as well. For a field size of 780 and pixel size
    # of 0.78 um the artefact is 42 um, but when pixel size is 0.39 um the
    # artefact becomes 91 um for some reason.
    if(ind == 0 and umpx < 0.31):
        crop_sz = 91
    else:
        crop_sz = crop_sz_arr[ind]

    # Apply frame time scaling
    crop_sz = crop_sz*t_fac

    # Convert crop size in um to px
    crop_sz_px = int(crop_sz/umpx)

    return crop_sz_px

def crop_scan_artefacts(img, config):
    """Crop away the sides of the image corrupted by galvo-scanning artefacts."""
    crop_sz_px = get_scan_artefact_sz(config=config)
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
    #mask[mask < 1] = 0
    return mask


def proc_img(file_name=None, rng=None, gamma=None, ch=2, corr_fi=False):
    """Process an image for analysis and display.

    Obtain specified mapping range and gamma values, crop scan artefacts and
    correct field illumination.
    """
    data = read_bin_file(file_name)
    config = read_cfg(file_name)

    if rng is None:
        rng = get_cfg_range(config, chan_id=ch)

    if gamma is None:
        gamma = get_cfg_gamma(config, ch=ch)

    if gamma is None:
        gamma = 1

    data_type = get_data_type(config=config)
    if data_type == DataType.SingleImage or data_type == DataType.Average:
        if data_type == DataType.SingleImage:
            img = data[:, :, ch]

        if data_type == DataType.Average:
            img = get_chan_sum(data=data, config=config, chan=ch)

        # Convert image to volts for analog channels
        # Assuming channel range is +-10V, no offset and 16bits
        if ch in (0, 1):
            img = (img.astype('float')/2**16 - 0.5)*20

        img = crop_scan_artefacts(img, config)

        if corr_fi:
            print("Correcting field illumination...")
            img = corr_field_illum(img, facpwr=get_nl_ord(config, ch))

        if isnone(rng):
            rng = get_opt_map_rng(img=img, file_name=file_name)

    return [img, rng, gamma, data]

def make_mosaic_img(data=None, mask=None, ij=None, pad=0.02, remap=True, rng=None):
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

        mos[row_ofs : nr + row_ofs, col_ofs : nc + col_ofs] = np.fliplr(img)

    return mos

def get_opt_map_rng(img=None, data=None, file_name=None, mask=None, ij=None):
    """Get optimal mapping range for an image."""
    if isnone(file_name):
        print("Dataset file name has to be provided")
        return None

    print("Estimating optimal data mapping range to 1% saturation.")

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
        print("Range estimation for tiled images doesn't yet work")
        return None

        #print("Crating dummy mosaic...")
        # TODO: This does not require ij indices. Remove requirement. # pylint: disable=W0511
        #if isnone(data) or isnone(mask) or isnone(ij):
            #[data, mask, ij] = get_tiling_data(data=data, file_name=file_name)
        #img = make_mosaic_img(data, mask, ij, remap=False)

    print("Determining optimal mapping range...")
    rng = get_frac_sat_rng(img)

    print("Mapping range: [%d , %d[" %(rng[0], rng[1]))
    return rng

def make_image(img=None, data=None, file_name=None, rng=None, gamma=None,
               ch=2, corr_fi=True, cmap=None, cmap_sat=False):
    """Make an image for display."""
    if isnone(img) or isnone(data):
        [img, rng, gamma, data] = proc_img(
            file_name=file_name, rng=rng, gamma=gamma, ch=ch,
            corr_fi=corr_fi)

    config = read_cfg(file_name)

    data_type = get_data_type(config=config)
    if data_type in (DataType.SingleImage, DataType.Average):
        img_raw = img

        if cmap_sat:
            map_cap = False
        else:
            map_cap = True

        [img, rng] = remap_img(img, rng=rng, gamma=gamma, cap=map_cap)
        img_scaled = img

        if isnone(cmap):
            cmap = get_def_chan_cmap(config)

        img = bake_cmap(img/255, cmap=cmap, remap=False, cm_over_val='r', cm_under_val='b')
    else:
        if data_type == DataType.TimeLapse:
            mos_type = MosaicType.TimeSeries
        elif data_type == DataType.ZStack:
            mos_type = MosaicType.ZStack
        else:
            print("Unknown data type" + str(data_type))

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

    mosaic = np.ndarray([nr*n_mr + (n_mr-1)*pad, nc*n_mc + (n_mc-1)*pad, 4], dtype='uint8')
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
        mosaic[from_r : to_r, from_c : to_c, :] = img*255

        image_coords[ind_mos, :] = [from_r, from_c]

        ind_mc = ind_mc + 1
        if ind_mc == n_mc:
            ind_mr += 1
            ind_mc = 0

    return [mosaic, image_coords]

def show_mosaic(data, file_name, mos_type=None, aspect=16/9, index_mask=None, det_ch=2):
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

    for ind in range(image_coords.shape[0]): # pylint: disable=E1136  # pylint/issues/3139
        cap_str = str(lbl[ind])
        if ind == 0:
            cap_str = label_str_pre + cap_str + label_str_suf
        plt.text(
            image_coords[ind, 1] + nc/2, image_coords[ind, 0] - 7, cap_str,
            horizontalalignment='center')

def make_composite_img(file_names, method="CombineToRGB", ofs=None, chas=None, corr_fi=True):
    """Make a composite RGB image."""
    if method == "CombineToRGB":
        nch = len(file_names)
        for ind in range(0, nch):

            data = make_image(file_names[ind], ch=2, corr_fi=corr_fi)

            if ind == 0:
                [nr, nc] = data[2].shape
                img = np.ndarray([nr, nc, 3]) # RGB output image
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
        #I0 = (D0[0])[:,:,0:3]
        #a0 = D0[1].astype(float)
        #I1 = (D1[0])[:,:,0:3]
        #a1 = D1[1].astype(float)

        #a0 = a0/a0.max()
        #a1 = a1/a1.max()

        #a0 = a0**0.5

        #a0 = a0/(a0+a1)
        #a1 = a1/(a0+a1)

        #I = .alpha_blend(I0, I1, a1=a0, a2=a1)

        #scipy.misc.imsave('I0.png', I0)
        #scipy.misc.imsave('I1.png', I1)
        #scipy.misc.imsave('I.png', I)

        return img
    else:
        print("Unknown method" + method)

    return None
