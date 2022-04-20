"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains routines for report generation.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import re
from shutil import copyfile
import platform

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import curve_fit
import configparser as cfg

from lklib.util import isnone, isarray, handle_general_exception, get_color, \
    printmsg
from lklib.fileread import list_files_with_filter, rem_extension, \
    change_extension, list_files_with_extension
from lklib.cfgparse import read_cfg
from lklib.image import remap_img, show_img, add_scale_bar, get_colourmap, \
     get_img_sz, save_img
from lklib.plot import export_figure
from lklib.trace import trace_set_param, get_pharos_log_trace
from lklib.report import MakeSVGReport, ConvertSVGToPDF
from lklib.fit import fit_gaussian_1d

from lcmicro.common import DataType, DetectorType, CountImageStats, \
    VoltageImageStats
from lcmicro.proc import make_image, make_composite_img, get_sat_mask, \
    get_opt_map_rng, proc_img
from lcmicro.cfgparse import get_sample_name, get_chan_name, get_laser_name, \
    get_ex_wavl, get_ex_power, get_def_chan_idx, get_chan_det_type, \
    get_chan_units, get_scan_frame_time, get_px_time, get_scan_field_size, \
    get_tiling_step, get_scan_px_sz, get_scan_date, get_operator_name, \
    get_sampe_id, get_sample_area_label, validate_chan_idx, get_cfg_range, \
    get_data_type, print_chan_name, get_microscope_name, \
    get_scan_field_size_calib_flag, print_data_info
from lcmicro.tiling import get_tiling_grid_sz, get_tiling_data, \
    show_raw_tiled_img, tile_images
from lcmicro.stab import get_stab_traces


def make_img_title(config, template="fig", chan=None, print_exw=False,
                   chas=None):
    """Make an image title string."""
    chan_name_str = None

    sample_name = get_sample_name(config)
    if not isnone(chan):
        chan_name_str = get_chan_name(config, chan)

    laser_name = get_laser_name(config)
    if not isnone(laser_name):
        wavl = get_ex_wavl(config)
        pwr = get_ex_power(config)

    if template == "fig":
        title_str = sample_name

        if print_exw and not isnone(laser_name):
            if not isnone(wavl) or not isnone(pwr):
                title_str = title_str + ", Ex."
            if not isnone(wavl):
                title_str = title_str + " {:.2f} µm".format(wavl)
            if not isnone(pwr):
                title_str = title_str + " {:.1f} mW".format(pwr)
        if chan is not None:
            if isarray(chan_name_str):
                chan_pre = ["R: ", "G: ", "B: "]

                str2 = ''
                for ind in enumerate(chan_name_str):
                    if not isnone(chas):
                        ch_ind = chas[ind]
                    else:
                        ch_ind = ind
                    str2 = str2 + chan_pre[ch_ind] + chan_name_str[ind] + '; '
                chan_name_str = str2

            if chan_name_str is not None:
                title_str = title_str + ", " + chan_name_str

    elif template == "report":
        title_str = sample_name + '\n'

        if not isnone(chan_name_str):
            title_str = title_str + chan_name_str

        if not isnone(wavl):
            title_str = title_str + ', Ex. ' + str(wavl) + ' um'
    else:
        print("Unsupported title template ''{:s}''".format(template))
        title_str = None

    return title_str


def make_caption_str(
        config, template="fig", ch_ind=None, rng=None, gamma=None, cmap=None,
        scalebar_sz=None, image_stats=None, img_sz=None):
    """Make a caption string for the figure."""
    if isnone(ch_ind):
        ch_ind = get_def_chan_idx

    ch_type = get_chan_det_type(config, ch_ind)

    if isarray(config):
        caption_str = ''
        numd = len(config)
        chan_pre = ["R: ", "G: ", "B: "]
        for indd in range(numd):
            caption_str = caption_str + chan_pre[indd] + make_caption_str(
                config[indd], rng=rng[indd], gamma=gamma[indd]) + '\n'

        caption_str = caption_str + 'bar = ' + str(scalebar_sz) + " um "

        return caption_str

    caption_str = ''

    caption_str = caption_str + "Ch: {:d}".format(ch_ind)

    if rng is not None:
        if ch_type == DetectorType.Counter:
            rng_str = "[{:d}, {:d}]".format(rng[0], rng[1])
        elif ch_type == DetectorType.Voltage:
            rng_str = "[{:.1f}, {:.1f}]".format(rng[0], rng[1])
        caption_str = caption_str + \
            ", range: {:s} {:s}".format(rng_str, get_chan_units(ch_type))

    if rng is not None:
        if gamma == 1:
            caption_str = caption_str + ", gamma: 1"
        else:
            caption_str = caption_str + ", gamma = {:1.1f}".format(gamma)

    if cmap is not None:
        caption_str = caption_str + ", cmap: " + cmap

    if template == "fig" and scalebar_sz is not None:
        caption_str = caption_str + ", bar = " + str(scalebar_sz) + " um"

    if ch_type == DetectorType.Counter:
        if image_stats.TotalCount is not None:
            frame_t = get_scan_frame_time(config)
            caption_str = caption_str + "\nAvg: {:.2f} Mcps".format(
                image_stats.TotalCount/frame_t/1E6)
        if image_stats.MaxCount is not None:
            px_t = get_px_time(config)
            caption_str = caption_str + ", max = {:.2f} Mcps".format(
                image_stats.MaxCount/px_t/1E6)
    elif ch_type == DetectorType.Voltage:
        if not isnone(image_stats.MinLevel):
            caption_str = caption_str + "\nMin: {:.2f} V".format(
                image_stats.MinLevel)
        if not isnone(image_stats.AvgLevel):
            caption_str = caption_str + ", avg: {:.2f} V".format(
                image_stats.AvgLevel)
        if not isnone(image_stats.MaxLevel):
            caption_str = caption_str + ", max: {:.2f} V".format(
                image_stats.MaxLevel)

    if template == "report":
        caption_str = caption_str + '\n'
        caption_str = caption_str + 'Tiling: '

        field_sz = get_scan_field_size(config, apply_sz_calib=False)
        if not isnone(field_sz):
            caption_str = caption_str + str(field_sz) + ' um size'

        tiling_grid_sz = get_tiling_grid_sz(config=config)
        if not isnone(tiling_grid_sz):
            caption_str = caption_str + ', {:d}x{:d} grid'.format(
                tiling_grid_sz[0], tiling_grid_sz[1])

        tiling_step_sz = get_tiling_step(config)
        if not isnone(tiling_step_sz):
            caption_str += ', {:.1f} mm step'.format(tiling_step_sz)

        pixel_sz = get_scan_px_sz(config, apply_sz_calib=False)
        if not isnone(pixel_sz):
            caption_str += ', pixel size: {:.2f} um'.format(pixel_sz)

        scan_area_x = img_sz[1] * pixel_sz
        scan_area_y = img_sz[0] * pixel_sz
        if not isnone(scan_area_x) and not isnone(scan_area_y):
            caption_str += ', scan area: {:.2f}x{:.2f} mm'.format(
                scan_area_x/1E3, scan_area_y/1E3)

        image_num_mpx = img_sz[0]*img_sz[1]
        if not isnone(image_num_mpx):
            caption_str += ', {:.1f} Mpx'.format(image_num_mpx/1E6)

        caption_str += '\n'

        date = get_scan_date(config)
        if not isnone(date):
            caption_str += 'Data: ' + date

        operator = get_operator_name(config)
        if not isnone(operator):
            caption_str += ', Scanned by: ' + operator

        sample_id = get_sampe_id(config)
        if not isnone(sample_id):
            caption_str += ', Sample: ' + sample_id

        sample_area_label = get_sample_area_label(config)
        if not isnone(sample_area_label):
            caption_str += ', Area ' + sample_area_label

    return caption_str


def export_img_png_tiff(
        file_name=None, verbosity='info', chan_ind=None, **kwargs):
    """Export microscopy data as a PNG or TIFF file."""
    if platform.architecture()[0] != '64bit':
        print("Running in 32-bit mode will likely fail for datasets larger "
              "than 500 MB.\nConsider installing 64-bit Python.")

    config = read_cfg(file_name)

    if verbosity == 'info':
        print("Processing file {:s}".format(file_name))
        print("Data info:")
        print_data_info(config, preffix='\t')
        print("\n")

    if isnone(chan_ind):
        print("Channel index not specified, looking for SHG channel...")
        chan_ind = get_def_chan_idx(config)
        print("SHG channel found at ch_ind={:d}".format(
            chan_ind))

    validate_chan_idx(config, chan_ind)
    chan_name = get_chan_name(config, chan_ind)
    print_chan_name(config, chan_ind)

    if isnone(config):
        print("Could not obtain config data, cannot generate image.")
        return False

    img, raw_img = make_image(
        file_name=file_name, ch=chan_ind, verbosity=verbosity, **kwargs)[0:2]

    # Blank top-left pixel, it usually contains garbge data
    raw_img[0, 0] = 0

    if np.max(raw_img) > 2**16:
        print("Data does not fit into 16 bits, truncating will occur")

    img_file_name = rem_extension(file_name) + chan_name + '.png'
    print("Writing '{:s}'".format(img_file_name))
    img2 = np.round(img[:, :, 0:3]*255).astype('uint8')
    cv2.imwrite(img_file_name, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    raw_img_file_name = rem_extension(file_name) + chan_name + '.tiff'
    print("Writing '{:s}'".format(raw_img_file_name))
    cv2.imwrite(raw_img_file_name, raw_img.astype('uint16'))
    print("All done")


def make_mosaic_fig(data=None, mask=None, ij=None, pad=0.02, rng=None):
    """Make a mosaic figure of images arranged by row and column indices.

    The row and column indices are given in ``ij``.

    This doesn't work well because of automatic figure scaling which results in
    different horizontal and vertical pixel spacing even though wspace and
    hspace are the same.
    """
    if isnone(rng):
        rng = [0, 20]
    num_grid_rows = ij[:, 0].max() + 1
    num_grid_cols = ij[:, 1].max() + 1

    grid = plt.GridSpec(num_grid_rows, num_grid_cols, wspace=pad, hspace=pad)

    for indt in enumerate(num_grid_rows*num_grid_cols):
        ax = plt.subplot(grid[ij[indt, 0], ij[indt, 1]])
        ax.set_aspect('equal')
        img = remap_img(data[:, :, mask[indt]], rng=rng)[0]
        img = np.fliplr(img)
        plt.imshow(img)
        plt.axis('off')


def gen_img_report(
        img=None, data=None, file_name=None, rng=None, chan_ind=None,
        gamma=None, chas=None, plot_raw_hist=True, plot_mapped_hist=True,
        plot_sat_map=True, do_export_figure=True, fig_suffix='', corr_fi=False,
        cmap=None, cm_sat=False, write_image=True, export_log_image=True,
        write_unprocessed_grayscale=False, verbosity='info', **kwargs):
    """Generate a report figure for an image.

    For the report generation to work correctly the header file should include
    the 'Microscope' name in the 'Setup' section. For the scan artefact removal
    to work correctly the artefact sizes should be added in
    get_scan_artefact_sz() and the 'Scan field calib valid' field be true in
    the 'Calibration' section.

    Additional parameters:
        crop_artefacts (bool) – remove scan flyback artefacts
    """
    config = read_cfg(file_name)

    if config is None:
        raise(Exception("Config file for '{:s}' not found".format(file_name)))

    if verbosity == 'info':
        print("Processing file {:s}".format(file_name))
        print("Options:")
        print("\tExport image: {:s}".format(str(write_image)))
        print("\tPlot raw histogram: {:s}".format(str(plot_raw_hist)))
        print("\tPlot mapped histogram: {:s}".format(str(plot_mapped_hist)))
        print("\tPlot saturation map: {:s}".format(str(plot_sat_map)))
        print("Data info:")
        print_data_info(config, preffix='\t')
        print("\n")

    if isnone(chan_ind):
        print("Channel index not specified, looking for SHG channel...")
        chan_ind = get_def_chan_idx(config)
        print("SHG channel found at ch_ind={:d}".format(
            chan_ind))

    validate_chan_idx(config, chan_ind)
    print_chan_name(config, chan_ind)
    chan_type = get_chan_det_type(config, chan_ind)

    if isnone(config):
        print("Could not obtain config data, cannot generate image.")
        return False

    if isinstance(file_name, type(str())):
        composite = False
        title_str = make_img_title(config, chan=chan_ind, print_exw=True)

        [img, img_raw, img_scaled, cmap, rng, gamma] = make_image(
            img=img, data=data, file_name=file_name, rng=rng, gamma=gamma,
            ch=chan_ind, corr_fi=corr_fi, cmap=cmap, cmap_sat=cm_sat,
            verbosity=verbosity, **kwargs)

        if export_log_image:
            img_log = make_image(
                img=img_raw, file_name=file_name, rng=rng, gamma=gamma, cmap=cmap,
                cmap_sat=cm_sat, map_scale='log', verbosity=verbosity,
                **kwargs)[0]

    else:
        composite = True
        title_str = make_img_title(config, chan=chan_ind, print_exw=True,
                                   chas=chas)
        [img, img_raw, img_scaled, cmap, rng, gamma] = make_composite_img(
            file_name, ofs=[None, None, None], chas=chas)

    [img, scalebar_sz] = add_scale_bar(img, pxsz=get_scan_px_sz(config))

    if export_log_image:
        img_log = add_scale_bar(img_log, pxsz=get_scan_px_sz(config))[0]

    printmsg('Scale bar is {:.0f} µm'.format(scalebar_sz), 'info', verbosity)

    grid = plt.GridSpec(2, 4)
    if not plot_raw_hist and not plot_mapped_hist and not plot_sat_map:
        plt.subplot(grid[0:2, 0:4])
    else:
        plt.subplot(grid[0:2, 0:2])

    show_img(img, title=title_str, remap=False)

    if write_image:
        mpimg.imsave(file_name[:file_name.rfind('.')] + fig_suffix + 'img' + '.png', img)

    if write_unprocessed_grayscale:
        plt.imsave(file_name[:file_name.rfind('.')] + fig_suffix + 'img_u' + '.png',
                   img_raw, vmin=rng[0], vmax=rng[1], cmap="gray")

    if export_log_image:
        printmsg('Exporting logarithmic scale image...', 'info', verbosity)
        mpimg.imsave(file_name[:file_name.rfind('.')] + fig_suffix + 'log_img' + '.png', img_log)


    if chan_type == DetectorType.Counter:
        img_stats = CountImageStats()
        if composite:
            img_stats.TotalCount = np.empty_like(gamma)
            img_stats.MaxCount = np.empty_like(gamma)
            for indch in enumerate(config):
                img_stats.TotalCount[indch] = img_raw.sum()
                img_stats.MaxCount[indch] = img.max()
        else:
            img_stats.TotalCount = img_raw.sum()
            img_stats.MaxCount = img_raw.max()
    elif chan_type == DetectorType.Voltage:
        img_stats = VoltageImageStats()
        img_stats.MinLevel = np.min(img_raw)
        img_stats.AvgLevel = np.mean(img_raw)
        img_stats.MaxLevel = np.max(img_raw)

    caption_str = make_caption_str(
        config, ch_ind=chan_ind, rng=rng, gamma=gamma, cmap=cmap,
        scalebar_sz=scalebar_sz, image_stats=img_stats)

    nr = img.shape[0]
    plt.text(0, nr*1.02, caption_str, verticalalignment='top')

    if plot_raw_hist:
        print("Plotting raw histogram...")
        plt.subplot(grid[0, 2])
        plt.hist(img_raw.flatten(), bins=256, log=True)
        ax = plt.gca()
        ax.set_title("Raw histogram")

    if plot_mapped_hist:
        print("Plotting mapped histogram...")
        plt.subplot(grid[0, 3])
        plt.hist(img_scaled.flatten(), bins=256, log=True)
        ax = plt.gca()
        ax.set_title("Mapped histogram")

    if plot_sat_map and not composite:
        sat_mask = get_sat_mask(img_raw, config)
        if not isnone(sat_mask):
            plt.subplot(grid[1, 2])
            show_img(sat_mask/4, cmap=get_colourmap("GYOR_Nice"), remap=False)
            ax = plt.gca()
            ax.set_title("Saturation map")
            if not (sat_mask > 1).any():
                plt.text(0, sat_mask.shape[0]*1.05, "No saturation")
            else:
                sat1 = (sat_mask > 1).sum()/len(sat_mask.flatten())
                if sat1 > 0.001:
                    sat1_str = "{:.3f}".format(
                        (sat_mask > 1).sum()/len(sat_mask.flatten()))
                else:
                    sat1_str = str((sat_mask > 1).sum()) + " px"

                plt.text(0, sat_mask.shape[0]*1.05,
                         "Saturation: >1 " + sat1_str
                         + "; >2 " + "{:.3f}".format(
                             (sat_mask > 2).sum()/len(sat_mask.flatten()))
                         + "; >4 " + "{:.3f}".format(
                             (sat_mask > 4).sum()/len(sat_mask.flatten())))

    # TODO: Make image reporting work for time series
    # else:
    #    if D_type == lk.DataType.TimeLapse:
    #        mos_type = lk.MosaicType.TimeSeries
    #    elif D_type == lk.DataType.ZStack:
    #        mos_type = lk.MosaicType.ZStack
    #    else:
    #        print("Unknown data type" + str(D_type))
    #        return None
    # lk.show_mosaic(data, file_name, mos_type=mos_type)

    if do_export_figure:
        print("Exporting report...")
        if composite:
            export_figure(file_name[0], suffix=fig_suffix + "comb", resize=True)
        else:
            export_figure(file_name, suffix=fig_suffix, resize=True)

    print("All done\n")

    return True


def gen_img_reports(corr_fi=False, **args):
    ret_val = True
    for ind in [2,3]:
        print("Processing channel {:d}...".format(ind))
        ret_val = ret_val and \
            gen_img_report(chan_ind=ind, corr_fi=corr_fi, fig_suffix="_ch{:d}".format(ind), **args)

    return ret_val


def gen_out_imgs(
        file_name=None, data=None, step_sz=None, rng=None, rng_override=None,
        make_basic_report_fig=True, make_detailed_report_fig=True,
        write_grayscale_img=False):
    """Generate a set of output images."""
    try:
        config = read_cfg(file_name)
        dtype = get_data_type(config=config)
        if isnone(dtype):
            print("Could not determine data type")
            raise Exception("InvalidDataType")

        if dtype == DataType.Tiling:
            # TODO: scan field size calibration is out of date. Fix it.
            img_sz = get_scan_field_size(config, apply_sz_calib=False)
            img_sz = [img_sz, img_sz]

            if isnone(step_sz):
                step_sz = get_tiling_step(config)*1000
                step_sz = [step_sz, step_sz]

            [data, mask, ij] = get_tiling_data(file_name=file_name, data=data)

            # TODO: get range for tiled images
            # if isnone(rng):
            #     rng = get_opt_map_rng(data=data, file_name=file_name,
            #                           mask=mask, ij=ij)

            print("Making raw tiled image...")
            show_raw_tiled_img(file_name=file_name, data=data, rng=rng)

            tile_images(
                data=data, file_name=file_name, img_sz=img_sz,
                step_sz=step_sz, rng=rng, rng_override=rng_override)
        else:
            [img, rng, gamma, data] = proc_img(file_name=file_name)
            if make_detailed_report_fig:
                plt.figure(1)
                gen_img_report(
                    img=img, data=data, file_name=file_name,
                    fig_suffix="detailed_fig", corr_fi=False, rng=rng,
                    gamma=gamma)

            if make_basic_report_fig:
                plt.figure(2)
                gen_img_report(
                    img=img, data=data, file_name=file_name,
                    fig_suffix="basic_fig", plot_raw_hist=False, rng=rng,
                    gamma=gamma, plot_mapped_hist=False, corr_fi=False,
                    plot_sat_map=False)

            if write_grayscale_img:
                img_save = np.round((img - rng[0])/(rng[1]-rng[0])*255)
                img_save[img_save > 255] = 255
                save_img(
                    img_save.astype(np.uint8),
                    ImageName=rem_extension(file_name), suffix="bw",
                    img_type="png", cmap="gray")
    except:
        handle_general_exception(
            "Could not generate output for file " + file_name)


def gen_stab_report(
        data_dir, dtype=None, ylim_norm=None, show_png_ext=False,
        copy_fig_to_storage=False, **kwargs):
    """Generate a signal stability report figure."""
    try:
        # Add a trailing backslash to the directory path if it is not given
        if data_dir[-1] != '\\':
            data_dir = data_dir + '\\'

        args = {'T_start': kwargs.get('T_start'), 'T_dur': kwargs.get('T_dur')}

        # Read THG, avg. power and peak intensity traces
        if dtype == "THG_Avg_Peak":
            [thg, p_avg, i_avg] = get_stab_traces(
                data_dir, Scaled3=True, **args)

            trs = [thg, p_avg, i_avg]
            trace_set_param(trs, Y_norm=True, ylim=ylim_norm)
            fig_file_name = "THG_vs_avg_and_peak.png"

        elif dtype in ("OscPwr", "OscBarTemp", "OscBarV", "OscBarI", "Ophir",
                       "THG_vs_nearXY", "THG_vs_farXY"):
            [thg, p_avg, i_avg] = get_stab_traces(data_dir, **args)
            p_avg.title = "Average Power (diode)"
            p_avg.Y_norm = True
            p_avg.ylim = ylim_norm

        # Read Pharos power log traces
        if dtype == "OscPwr":
            t_ofs_ph = kwargs.get("t_ofs_ph")
            p_avg_osc = get_pharos_log_trace(
                DirName=data_dir, DataType=dtype, T_ofs=t_ofs_ph, **args)
            p_avg_osc.Y_norm = True
            p_avg_osc.ylim = ylim_norm
            trs = [p_avg_osc, p_avg]
            fig_file_name = "FLINT_vs_diode.png"
        elif dtype == "OscBarTemp":
            t_ofs_ph = kwargs.get("t_ofs_ph")
            temp_bar = get_pharos_log_trace(
                DirName=data_dir, DataType=dtype, T_ofs=t_ofs_ph, **args)
            trs = [temp_bar, p_avg]
            fig_file_name = "OscBarTemp_vs_diode.png"
        elif dtype == "OscBarV":
            t_ofs_ph = kwargs.get("t_ofs_ph")
            temp_bar = get_pharos_log_trace(
                DirName=data_dir, DataType=dtype, T_ofs=t_ofs_ph, **args)
            trs = [temp_bar, p_avg]
            fig_file_name = "OscBarV_vs_diode.png"
        elif dtype == "OscBarI":
            t_ofs_ph = kwargs.get("t_ofs_ph")
            temp_bar = get_pharos_log_trace(
                DirName=data_dir, DataType=dtype, T_ofs=t_ofs_ph, **args)
            trs = [temp_bar, p_avg]
            fig_file_name = "OscBarI_vs_diode.png"
        elif dtype == "Ophir":
            t_ofs_ophir = kwargs.get("t_ofs_ophir")
            p_avg_ophir = read_tdms_trace_ophir(
                DirName=data_dir, T_ofs=t_ofs_ophir, **args)
            p_avg_ophir.title = "Average Power (Ophir)"
            p_avg_ophir.Y_norm = True
            p_avg_ophir.ylim = ylim_norm
            trs = [p_avg_ophir, p_avg]
            fig_file_name = "Ophir_vs_diode.png"

        # Read beam position traces
        if dtype in ("THG_vs_nearXY", "THG_vs_farXY", "beam_pos",
                     "beam_pos_ofs"):
            t_ofs_lcbd = kwargs.get("t_ofs_lcbd")

            data = read_lc_beam_diag(get_lc_beam_diag_path(data_dir, 2))
            near_x = Trace(T=data[:, 0], Y=data[:, 1],
                           title='Near Deviation X')
            near_y = Trace(T=data[:, 0], Y=data[:, 2],
                           title='Near Deviation Y')

            data = read_lc_beam_diag(get_lc_beam_diag_path(data_dir, 1))
            far_x = Trace(T=data[:, 0], Y=data[:, 1],
                          title='Far Deviation X')
            far_y = Trace(T=data[:, 0], Y=data[:, 2],
                          title='Far Deviation Y')

        # Make beam position stability plots
        if dtype == "THG_vs_nearXY":
            near_x.title = "Near Deviation X"
            near_y.title = "Near Deviation Y"
            trace_set_param(
                [near_x, near_y], T_ofs=t_ofs_lcbd, T_scale=1, sub_mean_y=True,
                Y_label='Deviation', Y_units='um', data_type='c', ref_val=0,
                ylim=[-30, 30], **args)
        elif dtype == "THG_vs_farXY":
            far_x.title = "Far Deviation X"
            far_y.title = "Far Deviation Y"
            trace_set_param(
                [far_x, far_y], T_ofs=t_ofs_lcbd, T_scale=1, sub_mean_y=True,
                Y_label='Deviation', Y_units='um', data_type='c', ref_val=0,
                ylim=[-30, 30], **args)

            trs = [thg, far_x, far_y]
            fig_file_name = "THG_vs_farXY.png"

        elif dtype == "beam_pos":
            near_x.title = "Near Position X"
            near_y.title = "Near Position Y"

            far_x.title = "Far Position X"
            far_y.title = "Far Position Y"

            trace_set_param(
                [near_x, near_y, far_x, far_y], T_ofs=t_ofs_lcbd, T_scale=1,
                Y_label='Position', Y_units='um', data_type='c', **args)

            trs = [near_x, near_y, far_x, far_y]
            fig_file_name = "beam_pos.png"

        elif dtype == "beam_pos_ofs":
            near_x.title = "Near Offset X"
            near_y.title = "Near Offset Y"
            near_x.ref_val = 2784
            near_y.ref_val = 1610

            far_x.title = "Far Offset X"
            far_y.title = "Far Offset Y"
            far_x.ref_val = 2410
            far_y.ref_val = 1529

            trace_set_param(
                [near_x, near_y, far_x, far_y], T_ofs=t_ofs_lcbd, T_scale=1,
                sub_ref_val=True, Y_label='Offset', Y_units='um',
                data_type='c', **args)

            trs = [near_x, near_y, far_x, far_y]
            fig_file_name = "beam_pos_ofs.png"

        compare_traces(
            trs=trs, show_stab_stats=True, plot_exp_sd=False, show_hist=False,
            **kwargs)

        fig_file_path = data_dir + fig_file_name
        export_figure(fig_file_path)

        if show_png_ext:
            show_png_ext(fig_file_path)

        if copy_fig_to_storage:
            copy_stab_fig_to_storage(data_dir, fig_file_path)

    except:
        handle_general_exception("Could not analyze trace")


def copy_stab_fig_to_storage(data_dir, fig_file_name):
    """Copy the stability report figure to the repostitory."""
    s = data_dir
    date_str = re.findall(r"(\d{4}-\d{2}-\d{2})", s)[0]
    dst = r"Z:\Projects\LCM\Data\Signal Stability\Stability Traces\\" \
        + date_str + ".png"
    copyfile(fig_file_name, dst)


def gen_report(file_name=None, img_file_names=None, chan_id=2, dry_run=False):
    """Generate a report."""
    if isnone(file_name):
        raise ValueError("File name not given")

    if isnone(img_file_names):
        img_file_names = list_files_with_filter(
            rem_extension(file_name) + 'Tiled_*' + '*.png')

    for img_file_name in img_file_names:
        img_sz = get_img_sz(file_name=img_file_name)

        if img_file_name.find("viridis") != -1:
            img_cmap = "viridis"
        elif(img_file_name.find("WK") != -1 or
             img_file_name.find("Greys") != -1):
            img_cmap = "WK"
        else:
            img_cmap = img_file_name[img_file_name.rfind('_')+1:-4]
            print("Unknown colourmap for file ''{:s}''".format(img_file_name)
                  + ", using ''{:s}''.".format(img_cmap))

        config = read_cfg(file_name)

        sample_id_str = get_sampe_id(config)
        sample_area_label = get_sample_area_label(config)

        sample_map_file_name = sample_id_str + '.svg'

        sample_map_href = sample_map_file_name + '#HE'
        sample_map_area_href = sample_map_file_name + '#' + sample_area_label

        rng = get_cfg_range(config, chan_id=chan_id)

        if isnone(rng):
            rng = get_opt_map_rng(file_name=file_name)

        um_px = get_scan_px_sz(config, apply_sz_calib=False)

        title_str = make_img_title(config, template="report", chan=chan_id)

        cap_str = make_caption_str(
            config, template="report", rng=rng, gamma=1, cmap=img_cmap,
            img_sz=img_sz)

        MakeSVGReport(
            img_file_name=img_file_name, img_sz=img_sz, um_px=um_px,
            img_cmap=img_cmap, title_str=title_str, cap_str=cap_str,
            sample_map_href=sample_map_href,
            sample_map_area_href=sample_map_area_href,
            dry_run=dry_run)

        ConvertSVGToPDF(change_extension(img_file_name, "svg"),
                        dry_run=dry_run)

def gen_thg_psf_fig(file_name=None, type='distal', wavl=None, suptitle_suffix=None, **kwargs):
    """Generate THG PSF report figure."""
    data = np.loadtxt(file_name)

    # The first point in a PSF scan might show a spurious point if the stage
    # did a flyback movement to the scan start position. Since the scan can be
    # done in any direction, the spurious point might be at the beginning
    # or the end, but it should be the first point in the array.
    data = data[1:, :]

    # Get z position in um and amplitude in Mcnt
    zpos = data[:, 0]*1E3
    ampl = data[:, 1]/1E6

    xlabel = 'Position, µm'
    ylabel = 'THG, Mcnt'

    if type is 'through':
        plt.figure(figsize=[5, 5])
        plt.plot(zpos, ampl, c=get_color('db'))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid('on')
    else:
        plt.figure(figsize=[10, 5])
        grid = plt.GridSpec(5, 2, wspace=0.1, hspace=0.1)
        main_lin_axes = plt.subplot(grid[0:4, 0])
        res_axes = plt.subplot(grid[4, 0])
        main_log_axes = plt.subplot(grid[0:4, 1])

        fit_gaussian_1d(
            zpos, ampl, plot=True, y_scale='lin',
            main_axes=main_lin_axes, res_axes=res_axes, center_z_axis_in_plot=True,
            plot_residuals=True, plot_fwhm=True,
            xlabel=xlabel, ylabel=ylabel, xlim=[-25, 25], **kwargs)

        fit_gaussian_1d(
            zpos, ampl, plot=True, y_scale='log', y_axis_pos='right',
            main_axes=main_log_axes, center_z_axis_in_plot=True,
            plot_residuals=False, plot_fwhm=True,
            xlabel=xlabel, ylabel=ylabel)

    if suptitle_suffix is None:
        if type is 'proximal':
            suptitle_suffix = 'proximal surface'
        elif type is 'distal':
            suptitle_suffix = 'distal surface'
        elif type is 'through':
            suptitle_suffix = 'through scan'

    suptitle_str = 'THG PSF'
    if wavl:
        if wavl < 0.2 or wavl > 20:
            print("Wavelength should be in microns")
        suptitle_str += ', λ={:.2f} µm'.format(wavl)

    if suptitle_suffix is not None:
        suptitle_str += ', ' + suptitle_suffix
    plt.suptitle(suptitle_str)

    export_figure(file_name, suffix='_THG_PSF', resize=False)


def pol_attn_func(x, A, T, x0, y0):
    """Polarizer attenuation function."""
    return A*np.cos((x-x0)/T) + y0


def calib_laser_power(file_name=None, trim_pts=2):
    """Get power attenuator calibration coefficients.

    Determine the calibration coefficients to convert from attenuator stepper
    motor position to transmitted power.

    Args:
        file_name (str): Name of calibration data file. Use the first text file
        in the current directory if not specified.
        timp_pts (int): Number of points to trim from the beginning and the end
            of the dataset.
    """
    if file_name is None:
        file_names = list_files_with_extension(ext='txt')
        if len(file_names) == 1:
            file_name = file_names[0]
        else:
            file_name = file_names[0]
            print("Multiple text files found, using the first one")

    print("Loading {:s} calibration file...".format(file_name))
    D = np.loadtxt(file_name)
    print("Done")

    M = D[trim_pts : -trim_pts, 0]
    P = D[trim_pts : -trim_pts, 1]

    plt.plot(M, P*1000, marker='.', c=get_color('db'))

    A_g = 0.06
    T_g = 3700
    x0_g = 12000
    y0_g = 0.06

    P_g = pol_attn_func(M, A_g, T_g, x0_g, y0_g)
    plt.plot(M, P_g*1000, c=get_color('dg'))

    popt = curve_fit(pol_attn_func, M, P, p0 = [A_g, T_g, x0_g , y0_g])[0]

    A_f = popt[0]
    T_f = popt[1]
    x0_f = popt[2]
    y0_f = popt[3]

    mpos_fit = np.linspace(np.min(M), np.max(M), 1000)
    power_fit = pol_attn_func(mpos_fit, A_f, T_f, x0_f, y0_f)

    plt.plot(mpos_fit, power_fit*1000, c=get_color('dr'))

    print('Fit model parameters:')
    print('A = {:.3f}, T = {:.0f}, x0 = {:.0f}, y0 = {:3f}'.format(A_f, T_f, x0_f, y0_f))

    Ap = 2*A_f
    R = np.pi * T_f
    Mofs = x0_f - R
    Pofs = y0_f - A_f

    if Mofs < 0:
        Mofs = Mofs + 2*R

    min_power = Pofs
    max_power = y0_f+A_f
    power_rng = 10*np.log10((y0_f + A_f)/(y0_f - A_f))
    period = R
    zero_offset = Mofs

    print('Attenuator model parameters:')
    print('Ap = {:.3f}, R = {:.0f}, Mofs = {:.0f}, Pofs = {:.3}'.format(Ap, R, Mofs, Pofs))
    print('Min power: {:.1f} mW, Max power: {:.0f} mW, Dynamic range: {:.0f} dB'.format(min_power*1000, max_power*1000, power_rng))
    print('Motor period: {:.0f} steps, zero offset: {:.0f} steps'.format(period, zero_offset))

    plt.legend(['Data', 'Guess', 'Fit'])
    plt.xlabel('Motor position, steps')
    plt.ylabel('Power, mW')
    plt.title('Power calibration')
    plt.grid('on')
    export_figure('pwrcalib.png', resize=False)

    config = cfg.RawConfigParser()
    calib_sec_str = 'Attenuator Calib'
    config.add_section(calib_sec_str)
    config.set(calib_sec_str, 'Ap', Ap)
    config.set(calib_sec_str, 'R', R)
    config.set(calib_sec_str, 'Mofs', Mofs)
    config.set(calib_sec_str, 'Pofs', Pofs)

    f = open('pwrcalib.ini', 'w')
    config.write( f )
    f.close()

    plt.show(block=True)

