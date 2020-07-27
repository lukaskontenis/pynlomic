"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains routines for signal stability analysis.

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from lklib.util import handle_general_exception, isarray, isnone
from lklib.fileread import read_big_file, read_starlab_file, read_tdms_traces
from lklib.cfgparse import read_cfg, get_head_val
from lklib.trace import Trace, trace_set_param


def read_stability_trace(file_name):
    """Read stability trace."""
    if isarray(file_name):
        file_names = file_name
        for ind, file_name in enumerate(file_names):
            trc_time, trc_data = read_stability_trace(file_name)

            if ind == 0:
                trc_time_arr = [trc_time]
                trc_data_arr = [trc_data]
            else:
                trc_time_arr.append(trc_time)
                trc_data_arr.append(trc_data)

        return [trc_time_arr, trc_data_arr]

    if file_name.find('SigMon') != -1:
        file_data = read_big_file(file_name)
    elif file_name.find('PowerLog'):
        file_data = read_starlab_file(file_name)
    else:
        print('Could not determine file type')
        return None

    trc_time = file_data[:, 0]
    trc_data = file_data[:, 1]

    # Handle overflows when time is reset to zero
    for i, t in enumerate(trc_time):
        if t == 0 and i != 0:
            t_step_pre_samples = trc_time[i-10:i-1] - trc_time[i-11:i-2]
            t_step_post_samples = trc_time[i:i+10] - trc_time[i+1:i+11]

            if t_step_pre_samples.std() > 1E-10:
                print("Pre samples spaced unequally")

            if t_step_post_samples.std() > 1E-10:
                print("Post samples spaced unequally")

            trc_time[i] = trc_time[i-1] + t_step_pre_samples.mean()*0.5 \
                + t_step_post_samples.mean()*0.5
            trc_time[i+1:] = trc_time[i+1:]+trc_time[i]

    return [trc_time - trc_time[0], trc_data]


def format_trace(
        trc_time, trc_data, t_ofs=None, t_scale=None, t_start=None,
        t_dur=None, baseline_corr=None, sub_mean_y=None):
    """Format trace."""
    if not isnone(t_ofs):
        trc_time = trc_time + t_ofs

    if not isnone(t_scale):
        trc_time = trc_time*t_scale

    if not isnone(t_start):
        ind = np.nonzero(trc_time > t_start)[0][0]
        trc_time = trc_time[ind:]
        trc_data = trc_data[ind:]

    if not isnone(t_dur):
        markers = np.nonzero(trc_time > trc_time[0] + t_dur)
        if len(markers[0]) > 0:
            ind = markers[0][0]
            trc_time = trc_time[0:ind]
            trc_data = trc_data[0:ind]

    if not isnone(baseline_corr):
        trc_data = trc_data/np.polyval(baseline_corr, trc_time)

    if sub_mean_y:
        trc_data = trc_data - trc_data.mean()

    return [trc_time, trc_data]


def plot_stability_trace(
        trc=None, trc_time=None, trc_data=None, reduce_data=True,
        file_name=None, descr=None, x_markers=None, plot_norm_trace=None,
        show_zero_line=True, plot_exp_sd=True, xlim=None, ylim=None,
        trace_ylim=None, hist_ylim=None, nf_ylim=None, data_type='Counts',
        title=None, show_x_label=True, ax_trace=None, ax_hist=None,
        ax_noisefac=None, ax_stats=None, show_trace=True, show_hist=True,
        show_noisefac=False, show_stats=False, show_stab_stats=None,
        **kwargs):
    """Plot a nice stability trace with data reduction and a hisgram.

    Show the envelope of all values in gray, the average trace in blue and the
    expected std. dev. bounds in red.

    TODO: This duplicates occurrence plot. Integrate the two.
    """
    try:
        if file_name is not None:
            [trc_time, trc_data] = read_stability_trace(file_name)

        if not isnone(trc):
            [trc_time, trc_data] = trc.GetTraceData(**kwargs)
            data_type = trc.data_type
            title = trc.title

            ylabel = trc.Y_label

            if isnone(plot_norm_trace):
                plot_norm_trace = trc.Y_norm

            if plot_norm_trace:
                ylabel = ylabel + ', a.u.'
            else:
                ylabel = ylabel + ', ' + trc.Y_units

        else:
            [trc_time, trc_data] = format_trace(trc_time, trc_data, **kwargs)

        if isnone(xlim):
            xlim = [min(trc_time), max(trc_time)]

        if isnone(plot_norm_trace):
            plot_norm_trace = False

        if isnone(ax_trace) and isnone(ax_hist) and isnone(ax_noisefac) \
                and isnone(ax_stats):
            if show_noisefac is False and show_stats is False:
                num_grid_rows = 1
            else:
                num_grid_rows = 2

            if show_hist is True:
                num_grid_cols = 3
            else:
                num_grid_cols = 2

            grid = plt.GridSpec(num_grid_rows, num_grid_cols,
                                wspace=0.5, hspace=0.5)

            ax_trace = plt.subplot(grid[0, 0:2])

            if show_hist:
                ax_hist = plt.subplot(grid[0, 2])

            if show_noisefac:
                ax_noisefac = plt.subplot(grid[1, 0:2])

            if show_stats:
                ax_stats = plt.subplot(grid[1, 2])

        trc_data_mean = trc_data.mean()

        # Determine expected standard deviation of the signal due to noise
        if data_type == 'd':
            # For discrete data (e.g. photon counts) use shot-noise which is
            # the square root of the mean
            data_exp_sd = np.sqrt(A_mean)
        elif data_type == 'c':
            # For continuous data (e.g volts) assume a fixed noise level
            data_exp_sd = 0.01

        if plot_norm_trace:
            data_plot = trc_data/A_mean
            data_mean_plot = 1
            data_exp_sd_plot = data_exp_sd/A_mean
        else:
            data_plot = trc_data
            data_mean_plot = A_mean
            data_exp_sd_plot = data_exp_sd

        if isnone(ylim) and not isnone(trc):
            if not isnone(trc.ylim):
                ylim = trc.ylim

        if not isnone(ylim):
            trace_ylim = ylim
            hist_ylim = ylim

        if show_trace:
            plt.axes(ax_trace)

            if show_zero_line:
                plt.plot(xlim, [0, 0], color=get_colour("black"))

            [plot_t, _, _] = plot_trace(
                trc_time, data_plot, marker='.', reduce_data=reduce_data)
            if plot_exp_sd:
                ym = data_mean_plot - data_exp_sd_plot
                plt.plot([plot_t[0], plot_t[-1]], [ym, ym],
                         color=get_colour("darkred"))

                ym = data_mean_plot + data_exp_sd_plot
                plt.plot([plot_t[0], plot_t[-1]], [ym, ym],
                         color=get_colour("darkred"))

            plt.ylim(trace_ylim)

            if not isnone(title):
                plt.title(title)
            else:
                plt.title("Stability trace")

            if show_x_label:
                plt.xlabel("Time (s)")

            plt.ylabel(ylabel)

            if not isnone(x_markers):
                for xm in x_markers:
                    plt.plot([xm, xm], plt.ylim(),
                             color=get_colour("darkgreen"))

            if not isnone(trc):
                if not isnone(trc.ref_val):
                    plt.plot(plt.xlim(), [trc.ref_val, trc.ref_val],
                             color=get_colour("gray"))

            plt.xlim(xlim)

            if show_stab_stats:
                # Print trace stability statistics in the upper left corner of
                # the plot axes

                # Add mean stability string, but only if the trace doesn't have
                # mean level substracton
                s = ''
                if not isnone(trc) and not trc.sub_mean_y:
                    s += 'Mean = {:.3f}'.format(data_plot.mean())

                # Format mean and std.dev. strings
                if not s == '':
                    s += ', '

                s += 'sd = {:.3f}'.format(data_plot.std())

                # Add fractional stability string, but only if the trace
                # doesn't have mean level subtraction
                if not isnone(trc) and not trc.sub_mean_y:
                    s += ', stab = {:.3f}'.format(
                        data_plot.std()/data_plot.mean())

                xlim = plt.xlim()
                ylim = plt.ylim()

                plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]),
                         ylim[1] - 0.02*(ylim[1]-ylim[0]),
                         s,
                         horizontalalignment='left', verticalalignment='top')

        if show_hist:
            plt.axes(ax_hist)

            if data_type == "Counts":
                bins = range(int(round(trc_data.min())),
                             int(round(trc_data.max())))
            elif data_type == "Volts":
                bins = np.arange(trc_data.min(), trc_data.max(), 0.00016)

            if plot_norm_trace:
                bins = bins/A_mean
                plt.hist(trc_data/A_mean, bins=bins, orientation="horizontal")
            else:
                plt.hist(trc_data, bins=bins, orientation="horizontal")

            plt.ylim(hist_ylim)

        if show_noisefac:
            plt.axes(ax_noisefac)

            # Bin data into 1 second bins
            [time_b, _, _] = reduce_trace(trc_time, trc_data,
                                          int((trc_time[-1] - trc_time[0])/1))

            if data_type == "Counts":
                noisefac = np.sqrt(A_mean)/Ib_sd
            elif data_type == "Volts":
                noisefac = 0.01/Ib_sd

            nf_m = noisefac.mean()
            nf_std = noisefac.std()

            plot_trace(time_b, noisefac, color=get_colour("darkblue"))
            plt.plot([time_b[0], time_b[-1]], [nf_m, nf_m],
                     color=get_colour("darkred"))
            plt.plot([time_b[0], time_b[-1]], [1, 1], color='k')
            plt.ylim(nf_ylim)
            plt.xlabel("Time (s)")
            plt.ylabel("Noise factor")

        if show_stats:
            plt.axes(ax_stats)
            plt.axis('off')

            if descr is None and file_name is not None:
                descr = get_head_val(read_cfg(file_name), "Scan Info",
                                     "Description")

            srate = 1/(trc_time[1] - trc_time[0])

            s = ''
            if descr is not None:
                s += descr + '\n'

            s += "Test duration: {:.1f} s\n".format(
                trc_time[-1] - trc_time[0]) + \
                "Sample rate: {:.1f} kHz\n".format(srate/1E3)

            if data_type == "Counts":
                s += "Mean count rate: {:.3f} Mcps\n".format(A_mean*srate/1E6)
            elif data_type == "Volts":
                s += "Mean signal: {:.3f} V\n".format(trc_data_mean)

            s += "Noise factor bin size: {:.3f} s\n".format(
                time_b[1] - time_b[0]) + \
                "Mean noise factor: {:.3f} ± {:.3f}\n".format(
                    nf_m, nf_std) + \
                "Min/max noise factor: {:.3f}, {:.3f}".format(
                    noisefac.min(), noisefac.max())

            plt.text(0, 1, s, verticalalignment='top')

    except:
        handle_general_exception("Could not plot trace")


def compare_traces(
        trc_arr=None, time_arr=None, data_arr=None, file_names=None,
        data_types=None, t_ofs_arr=None, T_scale_arr=None, **kwargs):
    """Compare traces."""
    trc = None
    file_name = None
    trc_time = None
    trc_data = None
    t_ofs = None
    t_scale = None
    data_type = None
    show_hist = True

    for key, value in kwargs.items():
        if key == 'show_hist':
            show_hist = value

    if not isnone(trc_arr):
        numr = len(trc_arr)
    elif not isnone(file_names):
        numr = len(file_names)
    else:
        numr = len(time_arr)

    if show_hist:
        grid = plt.GridSpec(numr, 3, wspace=0.5, hspace=0.5)
    else:
        grid = plt.GridSpec(numr, 2, wspace=0.5, hspace=0.5)

    for ind in range(0, numr):
        ax_trace = plt.subplot(grid[ind, 0:2])
        if show_hist:
            ax_hist = plt.subplot(grid[ind, 2])
        else:
            ax_hist = None

        if not isnone(trc_arr):
            trc = trc_arr[ind]

        if not isnone(file_names):
            file_name = file_names[ind]

        if not isnone(time_arr):
            trc_time = time_arr[ind]

        if not isnone(t_ofs_arr):
            t_ofs = t_ofs_arr[ind]

        if not isnone(T_scale_arr):
            t_scale = T_scale_arr[ind]

        if not isnone(data_arr):
            trc_data = data_arr[ind]

        if not isnone(data_types):
            data_type = data_types[ind]

        if ind == numr-1:
            show_x_label = True
        else:
            show_x_label = False

        plot_stability_trace(
            trc=trc, file_name=file_name, trc_time=trc_time, trc_data=trc_data,
            t_ofs=t_ofs, t_scale=t_scale, ax_trace=ax_trace, ax_hist=ax_hist,
            data_type=data_type, show_x_label=show_x_label, **kwargs)


def plot_comb_stab_traces(
        trc_time=None, trc_data=None, time_comb=None, data_comb=None,
        file_names=None, descr=None):
    """Plot combined stability traces."""
    t_splice = []
    if isnone(time_comb) and isnone(data_comb):
        numf = len(file_names)

        for indf, file_name in enumerate(file_names):
            print("Reading file " + str(indf))
            [trc_time, trc_data] = read_stability_trace(file_name)

            if indf == 0:
                time_comb = trc_time
                data_comb = trc_data
            else:
                t_splice.append(time_comb[-1])
                time_comb = np.append(time_comb, trc_time + time_comb[-1])
                data_comb = np.append(data_comb, trc_data)

    plot_stability_trace(time_comb, data_comb, descr=descr, X_markers=t_splice)


def get_stab_meas_start_t_and_dur(t_ofs_ts, data_dir=None):
    """Get the start time and duration of a stability measurement.

    The measurement is stored in a TDMS file,measurement start and stop times
    are expected to be stored as timestamps in Timing.ini.
    """
    # Read sync times
    if isnone(data_dir):
        data_dir = r".\\"

    timing_file = data_dir + r"\Timing.ini"
    cfg = read_cfg(timing_file)

    t_start_ts = time.mktime(datetime.strptime(get_head_val(
        cfg, 'Timing', 'Start Sync'), '%Y.%m.%d %H:%M:%S.%f').timetuple())

    t_end_ts = time.mktime(datetime.strptime(get_head_val(
        cfg, 'Timing', 'End Sync'), '%Y.%m.%d %H:%M:%S.%f').timetuple())

    # Read THG microscope signal, laser avg. power and avg. peak intensity
    # traces
    [_, _, t_ofs_ts] = read_tdms_traces(data_dir)

    t_ofs_ts = time.mktime(t_ofs_ts.timetuple())

    # Timing.ini and TDMS timestamps have different timezone handling. Subtract
    # two hours from both values to align.
    # Due to the delay in opening and closing the shutter even when using
    # blocking calls the trace is skewed in time with respect to sync
    # timestamps. Cut 1 s from the beginning of the trace and 3 s from the end
    # to make sure the shutter sync wiggles stay outside of the the stability
    # measurement range.
    t_start = t_start_ts - t_ofs_ts - 7200 + 1
    t_dur = t_end_ts - t_ofs_ts - 7200 - t_start - 3

    return [t_start, t_dur]


def get_stab_traces(dir_name, crop_t=True, t_start=None, t_dur=None,
                    scaled3=False):
    """Read nonlinear microscope stability traces.

    The stability traces are: THG from a glass-air surface, laser average
    power and laser avgerage peak intensity traces.
    """
    [trc_time, trc_data, tdms_ofs_ts] = read_tdms_traces(dir_name)

    [t_start_ts, t_dur_ts] = get_stab_meas_start_t_and_dur(
        tdms_ofs_ts, data_dir=dir_name)

    if isnone(t_start):
        t_start = t_start_ts

    if isnone(t_dur):
        t_dur = t_dur_ts

    thg = Trace(trc_time=trc_time[0], Y=trc_data[0], Y_label='Counts',
                title='THG', data_type='d')

    if scaled3:
        avg_pwr = Trace(trc_time=trc_time[0], Y=trc_data[1]**3, Y_label='Volts',
                      title='Avg. Power Ref. ^3', data_type='c')
        avg_intensity = Trace(trc_time=trc_time[0], Y=trc_data[2]**3/2, Y_label='Volts',
                      title='SHG Ref. ^3/2', data_type='c')
    else:
        avg_pwr = Trace(trc_time=trc_time[0], Y=trc_data[1], Y_label='Volts',
                      title='Avg. Power Ref.', data_type='c')
        avg_intensity = Trace(trc_time=trc_time[0], Y=trc_data[2], Y_label='Volts',
                      title='SHG Ref.', data_type='c')

    if crop_t:
        trace_set_param([thg, avg_pwr, avg_intensity], t_ofs=0, t_start=t_start,
                        t_dur=t_dur)

    return [thg, avg_pwr, avg_intensity]
