"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains routines for report generation.

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import re
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lklib.util import isnone, isarray, handle_general_exception
from lklib.fileread import list_files_with_filter, rem_extension, change_extension
from lklib.cfgparse import read_cfg
from lklib.image import remap_img, show_img, add_scale_bar, get_colourmap, get_img_sz, save_img
from lklib.plot import export_figure
from lklib.trace import trace_set_param, get_pharos_log_trace
from lklib.report import MakeSVGReport, ConvertSVGToPDF

from lcmicro.common import DataType, DetectorType, CountImageStats, VoltageImageStats
from lcmicro.proc import make_image, make_composite_img, get_sat_mask, \
    get_opt_map_rng, proc_img
from lcmicro.cfgparse import get_sample_name, get_chan_name, get_laser_name, \
    get_ex_wavl, get_ex_power, get_def_chan_idx, get_chan_det_type, \
    get_chan_units, get_scan_frame_time, get_px_time, get_scan_field_size, \
    get_tiling_step, get_scan_px_sz, get_scan_date, get_operator_name, \
    get_sampe_id, get_sample_area_label, validate_chan_idx, get_cfg_range, get_data_type
from lcmicro.tiling import get_tiling_grid_sz, get_tiling_data, show_raw_tiled_img, tile_images
from lcmicro.stab import get_stab_traces

def make_img_title(config, template="fig", chan=None, print_exw=False, chas=None):
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
                title_str = title_str + ", Ex. "
            if not isnone(wavl):
                title_str = title_str + "%.2f um" %(wavl)
            if not isnone(pwr):
                title_str = title_str + "%.1f mW" %(pwr)
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
        print("Unsupported title template ''%s''" %template)
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
            caption_str = caption_str + chan_pre[indd] \
            + make_caption_str(
                config[indd], rng=rng[indd], gamma=gamma[indd]) + '\n'

        caption_str = caption_str + 'bar = ' + str(scalebar_sz) + " um "

        return caption_str

    caption_str = ''

    caption_str = caption_str + "Ch: %d" %(ch_ind)

    if rng is not None:
        if ch_type == DetectorType.Counter:
            rng_str = "[%d, %d]" %(rng[0], rng[1])
        elif ch_type == DetectorType.Voltage:
            rng_str = "[%.1f, %.1f]" %(rng[0], rng[1])
        caption_str = caption_str + \
            ", range: %s %s" %(rng_str, get_chan_units(ch_type))

    if rng is not None:
        if gamma == 1:
            caption_str = caption_str + ", gamma: 1"
        else:
            caption_str = caption_str + ", gamma = %1.1f" % (gamma)

    if cmap is not None:
        caption_str = caption_str + ", cmap: " + cmap

    if template == "fig" and scalebar_sz is not None:
        caption_str = caption_str + ", bar = " + str(scalebar_sz) + " um"

    if ch_type == DetectorType.Counter:
        if image_stats.TotalCount is not None:
            frame_t = get_scan_frame_time(config)
            caption_str = caption_str + "\nAvg: %.2f Mcps" %(image_stats.TotalCount/frame_t/1E6)
        if image_stats.MaxCount is not None:
            px_t = get_px_time(config)
            caption_str = caption_str + ", max = %.2f Mcps" %(image_stats.MaxCount/px_t/1E6)
    elif ch_type == DetectorType.Voltage:
        if not isnone(image_stats.MinLevel):
            caption_str = caption_str + "\nMin: %.2f V" %(image_stats.MinLevel)
        if not isnone(image_stats.AvgLevel):
            caption_str = caption_str + ", avg: %.2f V" %(image_stats.AvgLevel)
        if not isnone(image_stats.MaxLevel):
            caption_str = caption_str + ", max: %.2f V" %(image_stats.MaxLevel)

    if template == "report":
        caption_str = caption_str + '\n'
        caption_str = caption_str + 'Tiling: '

        field_sz = get_scan_field_size(config, apply_sz_calib=False)
        if not isnone(field_sz):
            caption_str = caption_str + str(field_sz) + ' um size'

        tiling_grid_sz = get_tiling_grid_sz(config=config)
        if not isnone(tiling_grid_sz):
            caption_str = caption_str + ', %dx%d grid' %(tiling_grid_sz[0], tiling_grid_sz[1])

        tiling_step_sz = get_tiling_step(config)
        if not isnone(tiling_step_sz):
            caption_str = caption_str + ', %.1f mm step' %tiling_step_sz

        pixel_sz = get_scan_px_sz(config, apply_sz_calib=False)
        if not isnone(pixel_sz):
            caption_str = caption_str + ', pixel size: %.2f um' %pixel_sz

        scan_area_x = img_sz[1] * pixel_sz
        scan_area_y = img_sz[0] * pixel_sz
        if not isnone(scan_area_x) and not isnone(scan_area_y):
            caption_str = caption_str \
                + ', scan area: {:.2f}x{:.2f} mm'.format(
                    scan_area_x/1E3, scan_area_y/1E3)

        image_num_mpx = img_sz[0]*img_sz[1]
        if not isnone(image_num_mpx):
            caption_str = caption_str + ', %.1f Mpx' %(image_num_mpx/1E6)

        caption_str = caption_str + '\n'

        date = get_scan_date(config)
        if not isnone(date):
            caption_str = caption_str + 'Data: ' + date

        operator = get_operator_name(config)
        if not isnone(operator):
            caption_str = caption_str + ', Scanned by: ' + operator

        sample_id = get_sampe_id(config)
        if not isnone(sample_id):
            caption_str = caption_str + ', Sample: ' + sample_id

        sample_area_label = get_sample_area_label(config)
        if not isnone(sample_area_label):
            caption_str = caption_str + ', Area ' + sample_area_label

    return caption_str


def make_mosaic_fig(data=None, mask=None, ij=None, pad=0.02, rng=None):
    """Make a mosaic figure of images arranged according to row and column indices.

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
        #plt.imshow(I)
        img = remap_img(data[:, :, mask[indt]], rng=rng)[0]
        img = np.fliplr(img)
        plt.imshow(img)
        plt.axis('off')


def gen_img_report(
        img=None, data=None, file_name=None, rng=None, chan_ind=None,
        gamma=None, chas=None, plot_raw_hist=True, plot_mapped_hist=True,
        plot_sat_map=True, do_export_figure=True, fig_suffix='', corr_fi=True,
        cmap=None, cm_sat=False, write_image=True,
        write_unprocessed_grayscale=False):
    """Generate a report figure for an image."""
    config = read_cfg(file_name)

    if isnone(chan_ind):
        chan_ind = get_def_chan_idx(config)
        print("Channel index not specified assuming ch_ind=%d" %(chan_ind))

    validate_chan_idx(config, chan_ind)
    chan_type = get_chan_det_type(config, chan_ind)

    if isnone(config):
        print("Could not obtain config data, cannot generate image.")
        return

    if isinstance(file_name, type(str())):
        composite = False
        title_str = make_img_title(config, chan=chan_ind, print_exw=True)
        [img, img_raw, img_scaled, cmap, rng, gamma] = make_image(
            img=img, data=data, file_name=file_name, rng=rng, gamma=gamma,
            ch=chan_ind, corr_fi=corr_fi, cmap=cmap, cmap_sat=cm_sat)
    else:
        composite = True
        title_str = make_img_title(config, chan=chan_ind, print_exw=True, chas=chas)
        [img, img_raw, img_scaled, cmap, rng, gamma] = make_composite_img(
            file_name, ofs=[None, None, None], chas=chas)

    [img, scalebar_sz] = add_scale_bar(img, pxsz=get_scan_px_sz(config))

    grid = plt.GridSpec(2, 4)
    if not plot_raw_hist and not plot_mapped_hist and not plot_sat_map:
        plt.subplot(grid[0:2, 0:4])
    else:
        plt.subplot(grid[0:2, 0:2])

    show_img(img, title=title_str, remap=False)

    if write_image:
        mpimg.imsave(file_name[:file_name.rfind('.')] + 'img' + '.png', img)

    if write_unprocessed_grayscale:
        plt.imsave(file_name[:file_name.rfind('.')] + 'img_u' + '.png',
                   img_raw, vmin=rng[0], vmax=rng[1], cmap="gray")

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
        plt.subplot(grid[0, 2])
        plt.hist(img_raw.flatten(), bins=256, log=True)
        ax = plt.gca()
        ax.set_title("Raw histogram")

    if plot_mapped_hist:
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
                    sat1_str = "%.3f" %((sat_mask > 1).sum()/len(sat_mask.flatten()))
                else:
                    sat1_str = str((sat_mask > 1).sum()) + " px"

                plt.text(0, sat_mask.shape[0]*1.05,
                         "Saturation: >1 "+ sat1_str + "; "
                         ">2 "+ "%.3f" %((sat_mask > 2).sum()/len(sat_mask.flatten())) + "; "
                         #">3 "+ "%.3f" %((sat_mask>3).sum()/len(sat_mask.flatten())) + "; "
                         ">4 "+ "%.3f" %((sat_mask > 4).sum()/len(sat_mask.flatten())))

    #else:
    #    if D_type == lk.DataType.TimeLapse:
    #        mos_type = lk.MosaicType.TimeSeries
    #    elif D_type == lk.DataType.ZStack:
    #        mos_type = lk.MosaicType.ZStack
    #    else:
    #        print("Unknown data type" + str(D_type))
    #        #return None
    #
    #    lk.show_mosaic(data, file_name, mos_type=mos_type)

    if do_export_figure:
        if composite:
            export_figure(file_name[0], suffix=fig_suffix + "comb")
        else:
            export_figure(file_name, suffix=fig_suffix)


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
            # TODO: scan field size calibration is out of date. Fix it. # pylint: disable=W0511
            img_sz = get_scan_field_size(config, apply_sz_calib=False)
            img_sz = [img_sz, img_sz]

            if isnone(step_sz):
                step_sz = get_tiling_step(config)*1000
                step_sz = [step_sz, step_sz]

            [data, mask, ij] = get_tiling_data(file_name=file_name, data=data)

            if isnone(rng):
                rng = get_opt_map_rng(data=data, file_name=file_name, mask=mask, ij=ij)

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
    except: # pylint: disable=W0702
        handle_general_exception("Could not generate output images for file " + file_name)

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
        if dtype in ("THG_vs_nearXY", "THG_vs_farXY", "beam_pos", "beam_pos_ofs"):
            t_ofs_lcbd = kwargs.get("t_ofs_lcbd")

            data = read_lc_beam_diag(get_lc_beam_diag_path(data_dir, 2))
            near_x = Trace(T=data[:, 0], Y=data[:, 1], title='Near Deviation X')
            near_y = Trace(T=data[:, 0], Y=data[:, 2], title='Near Deviation Y')

            data = read_lc_beam_diag(get_lc_beam_diag_path(data_dir, 1))
            far_x = Trace(T=data[:, 0], Y=data[:, 1], title='Far Deviation X')
            far_y = Trace(T=data[:, 0], Y=data[:, 2], title='Far Deviation Y')

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

    except: # pylint: disable=W0702
        handle_general_exception("Could not analyze trace")


def copy_stab_fig_to_storage(data_dir, fig_file_name):
    """Copy the stability report figure to the repostitory."""
    s = data_dir
    date_str = re.findall(r"(\d{4}-\d{2}-\d{2})", s)[0]
    dst = r"Z:\Projects\LCM\Data\Signal Stability\Stability Traces\\" + date_str + ".png"
    copyfile(fig_file_name, dst)

def gen_report(file_name=None, img_file_names=None, chan_id=2, dry_run=False):
    """Generate a report."""
    if isnone(file_name):
        raise ValueError("File name not given")

    if isnone(img_file_names):
        img_file_names = list_files_with_filter(rem_extension(file_name) + 'Tiled_*' + '*.png')

    for img_file_name in img_file_names:
        img_sz = get_img_sz(file_name=img_file_name)

        if img_file_name.find("viridis") != -1:
            img_cmap = "viridis"
        elif(img_file_name.find("WK") != -1 or img_file_name.find("Greys") != -1):
            img_cmap = "WK"
        else:
            img_cmap = img_file_name[img_file_name.rfind('_')+1:-4]
            print("Unknown colourmap for file ''%s'', using ''%s''." %(img_file_name, img_cmap))

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

        ConvertSVGToPDF(change_extension(img_file_name, "svg"), dry_run=dry_run)
