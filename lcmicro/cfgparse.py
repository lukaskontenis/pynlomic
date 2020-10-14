
"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains routines to parse microscopy data metainformation and
configuration files.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
from datetime import datetime

import numpy as np

from lklib.util import isnone, isarray
from lklib.string import timestamp_str_to_seconds, make_human_time_str
from lklib.cfgparse import read_cfg, get_cfg_section, get_head_val

from lcmicro.common import DetectorType, PixelCountLimit, DataType, \
     get_data_type_str


def get_sample_name(config):
    """Get the name of the sample."""
    if isarray(config):
        return get_sample_name(config[0])
    return get_head_val(config, "Sample", "Name")


def get_sampe_id(config):
    """Get the ID of the sample."""
    return get_head_val(config, "Sample", "ID")


def get_sample_area_label(config):
    """Get the area label of the sample."""
    return get_head_val(config, "Sample", "Area")


def get_scan_date(config):
    """Get the scan date."""
    return get_head_val(config, "Scan Info", "Scan date")


def get_scan_descr(config):
    """Get the scan description string."""
    return get_head_val(config, "Scan Info", "Description")


def get_operator_name(config):
    """Get the operator name."""
    return get_head_val(config, "Scan Info", "Operator")


def get_microscope_name(config):
    """Get the microscope name."""
    return str(get_head_val(config, "Setup", "Microscope"))


def get_num_chan(config):
    """Get the number of channels in dataset."""
    for chan_id in range(255):
        if isnone(get_cfg_section(config, "Channel " + str(chan_id))):
            return chan_id
    raise Exception("InvalidChannelNumber")


def get_chan_name(config, chan):
    """Get the name of channel a channel."""
    if isarray(config):
        numd = len(config)
        names = []
        for indd in range(0, numd):
            names.append(get_chan_name(config[indd], chan))
        return names
    return get_head_val(config, "Channel " + str(chan), "Name")


def print_chan_name(config, ch_ind):
    print('Using channel {:d}, {:s}'.format(ch_ind, get_chan_name(config, ch_ind)))


def get_chan_hw_name(config, chan_ind):  # pylint: disable=W0613
    """Get the hardware name of the channel.

    Get the hardware source name string of the channel index. This information
    should be available in the config file.
    """
    hw_chan_map = {
        0: "AI0",
        1: "AI1",
        2: "CNT0",
        3: "CNT1"
    }
    return hw_chan_map.get(chan_ind, "invalid")


def get_chan_det_type(config, chan_ind):
    """Get the detector type of a channel.

    TODO: The function should determine the detector from the config file.
    """
    validate_chan_idx(config, chan_ind)
    if(chan_ind == 0 or chan_ind == 1):
        chan_type = DetectorType.Voltage
    elif(chan_ind == 2 or chan_ind == 3):
        chan_type = DetectorType.Counter
    return chan_type


def get_chan_units(chan_type):
    """Get the units of a channel type."""
    if chan_type == DetectorType.Voltage:
        chan_units = "V"  # Volts
    elif chan_type == DetectorType.Counter:
        chan_units = "c."  # Counts
    return chan_units


def get_def_chan_idx(config):  # pylint: disable=W0613
    """Get the default channel index."""
    return 2


def validate_chan_idx(config, chan_ind):  # pylint: disable=W0613
    """Validate the channel index.

    TODO: This should read the config file to check which channels are defined.
    """
    if chan_ind < 0 or chan_ind > 2:
        print("Unsupported channel index")
        return False
    else:
        return True


def get_nl_ord(config, chan):
    """Get the nonlinear order of a channel."""
    name = get_chan_name(config, chan)
    if name == "SHG":
        return 2
    elif name == "THG":
        return 3
    elif name[0:2] == "Ax":
        # Alexa MPEF
        return 2
    else:
        print("Unknown nonlinear order of channel '"
              + str(chan) + "', assuming 2")
        return 2
    return -1


def get_chan_filter_name(config, chan):
    """Get the spectral filter name of a channel."""
    name = get_head_val(config, "Channel " + str(chan), "Filter")
    if isnone(name):
        return None
    else:
        return name.strip(r'"')


def get_laser_name(config):
    """Get the laser name string."""
    return get_head_val(config, "Setup", "Laser")


def get_ex_wavl(config):
    """Get the excitation wavelength in um."""
    if isarray(config):
        return get_ex_wavl(config[0])

    exw_str = get_head_val(config, "Setup", "Laser wavelength")
    if exw_str is not None:
        try:
            exw = float(exw_str)
        except ValueError:
            print("Cannot parse laser wavelength")
            exw = None
        return exw
    else:
        return None


def get_ex_power(config):
    """Get the excitation power in W."""
    if isarray(config):
        return get_ex_power(config[0])

    pwr_str = get_head_val(config, "Setup", "Laser power")
    if pwr_str is not None:
        try:
            pwr = float(pwr_str)
        except ValueError:
            print("Cannot parse laser power")
            pwr = None
        return pwr
    else:
        return None


def get_det_sat_thr(config):  # pylint: disable=W0613
    """Get the count saturation threshold of the detector.

    TODO: This assumes H10682 photon counter. Should validate with the config
    file.
    """
    return 5E6


def get_px_cnt_limits(config):
    """Get all count limits."""
    pixel_t = get_px_time(config)
    rep_f = get_ex_rep_rate(config)
    sat_f = get_det_sat_thr(config)

    limit = PixelCountLimit()
    limit.RepRate = int(np.round(pixel_t*rep_f))
    limit.SinglePulse = int(np.round(pixel_t*rep_f*0.1))
    limit.CountLinearity = int(np.round(pixel_t*sat_f))
    return limit


def get_px_cnt_limit(config, ignore_linearity_limit=True):
    """Get the maximum expected number of counts per pixel."""
    limit = get_px_cnt_limits(config)
    if ignore_linearity_limit:
        return np.min([limit.RepRate, limit.SinglePulse])
    else:
        return np.min([limit.RepRate, limit.SinglePulse, limit.CountLinearity])


def get_px_bckgr_count(config):  # pylint: disable=W0613
    """Get the expected counts per pixel due to background.

    Using photon counters at µs dwell times results in at most 1 count per
    pixel due to dark counts. This then assumes a background level of 2 c./px
    """
    return 2


def get_stage_pos(config, axis=None, index=None):
    """Get the stage position on an axis.

    If ``index`` is specified its position is returned, otherwise the
    reference stage position is returned.
    """
    if isnone(index):
        pos_str = get_head_val(config, "Position", axis)
    else:
        pos_str = get_head_val(config, "Index " + str(index), axis)

    pos = None
    if pos_str is not None:
        try:
            pos = float(pos_str)
        except ValueError:
            print("Cannot parse stage position")
    return pos


def get_stage_xyz_pos(config=None, file_name=None, index=None):
    """Get the XYZ sample stage position of a data store entry."""
    if isarray(file_name):
        file_names = file_name

        pos = np.ndarray([len(file_names), 3])
        for ind, file_name in enumerate(file_names):
            pos[ind, :] = get_stage_xyz_pos(file_name=file_name)
        return pos

    if config is None and file_name is None:
        print("Config object or file name must be supplied")

    if isnone(config):
        config = read_cfg(file_name)

    X = get_stage_pos(config, "X", index=index)
    Y = get_stage_pos(config, "Y", index=index)
    Z = get_stage_pos(config, "Z", index=index)

    return [X, Y, Z]


def get_ex_rep_rate(config):
    """Get the repetition rate of the excitation source."""
    laser = get_laser_name(config)
    if isnone(laser):
        return None
    laser = laser.strip(r'"')

    if laser == "FLINT":
        return 76E6

    print("Cannot determine repetition rate for " + laser)
    return None


def get_scan_field_size(config, apply_sz_calib=True):
    """Get the scan field size in µm."""
    fieldsz_um = get_head_val(config, "Scan Geometry", "Field size")
    if fieldsz_um is None:
        fieldsz_um = get_head_val(config, "Scan", "Field size")
    if fieldsz_um is None:
        print("Cannot find scan field size in header")
        return None

    fieldsz_um = float(fieldsz_um)
    if apply_sz_calib:
        calib_corr = get_scan_field_calib_corr(config)
        fieldsz_um = fieldsz_um*calib_corr
    return fieldsz_um


def get_scan_frame_time(config):
    """Get the scan frame time in s."""
    frame_t_s = get_head_val(config, "Scan Geometry", "Frame time")
    if frame_t_s is None:
        frame_t_s = get_head_val(config, "Scan", "Frame time")
    if frame_t_s is None:
        print("Cannot find frame time in header")
        return None
    return float(frame_t_s)


def get_px_time(config):
    """Get pixel dwell time time in seconds."""
    frame_t_s = get_scan_frame_time(config)
    res = get_scan_resolution(config)
    if(frame_t_s is None or res is None):
        print("Cannot determine pixel time")
        return None
    return frame_t_s/res/res


def get_total_scan_time(config):
    """Get the total scan time in seconds.

    Scan time excludes stage movement and other overhead.
    """
    num_f = get_num_frames(config)
    frame_t = get_scan_frame_time(config)
    return num_f * frame_t


def get_total_meas_time(config):
    """Get the total measurement time in seconds."""
    num_idx = get_data_store_idx_len(config)
    return timestamp_str_to_seconds(get_idx_ts(config, num_idx-1)) \
        - timestamp_str_to_seconds(get_idx_ts(config, 0))


def get_scan_resolution(config):
    """Get the scan resolution in px."""
    nr = get_head_val(config, "Scan Geometry", "Lines")
    nc = get_head_val(config, "Scan Geometry", "Columns")

    if(nr is None or nc is None):
        nr = get_head_val(config, "Scan", "Lines")
        nc = get_head_val(config, "Scan", "Columns")

    if(nr is None or nc is None):
        print("Cannot find scan resolution in header")
        return None

    nr = float(nr)
    nc = float(nc)
    if nr != nc:
        print('Image is not square!')

    return (nr + nc)/2


def get_scan_field_size_calib_flag(config):
    return get_head_val(config, "Calibration", "Scan field calib valid")


def get_scan_field_size_calib_date(config):
    return get_head_val(config, "Calibration", "Scan field calib date")
    """Get calibration correction for a physical scan field size.

    This function should be used to correct previous scan data if the scan
    field calibration is later determined to be wrong.
    """
    scan_field_calib_date = None
    has_calib = False
    if 'Calibration' in config.sections():
        has_calib = True
        calibcfg = config['Calibration']
        scan_field_calib_date = calibcfg.get('Scan field calib date', None)

    # If no scan field calibration is defined or it is older than 2018.04.04,
    # the scan field size calibration is wrong and needs to be corrected
    if not has_calib or (has_calib and
                         scan_field_calib_date < datetime(2018, 4, 4)):
        return 0.785

    # Otherwise all is fine
    return 1.0


def get_scan_px_sz(config, **kwargs):
    """Get the size of the scan pixel in um."""
    if isarray(config):
        nd = len(config)
        umpx = np.ndarray(nd, float)
        for indd in range(0, nd):
            umpx[indd] = get_scan_px_sz(config[indd])

        if not (umpx.mean() == umpx).all():
            print("Unequal pixel sizes")

        return umpx.mean()

    field_sz_um = get_scan_field_size(config, **kwargs)
    img_res_col = get_scan_resolution(config)
    if(field_sz_um is None or img_res_col is None):
        print("Cannot determine pixel size")
        return None
    else:
        umpx = field_sz_um/img_res_col
        return umpx


def get_data_store_idx_len(config):
    """Get the length of the data store index."""
    num_sec = 0
    while 1:
        sec_name = 'Index ' + str(num_sec)
        if sec_name not in config.sections():
            return num_sec
        num_sec = num_sec + 1


def get_data_store_entry(config, ind):
    """Get the a data store entry."""
    sec_name = 'Index ' + str(ind)
    return config[sec_name]


def get_num_frames(config):
    """Get the number of frames."""
    num_idx = get_data_store_idx_len(config)
    num_ch = get_num_chan(config)

    num_f = num_idx/num_ch
    if num_f % 1 != 0:
        print("WARNING: Number of frames in not a round number")

    return int(num_f)


def get_idx_mask(config, chan_sel):
    """Get the data index mask."""
    nd = get_data_store_idx_len(config)
    indm = 0
    mask = np.ndarray(nd, dtype=int)

    chan_sel_str = get_chan_hw_name(config, chan_sel)
    if chan_sel_str == "invalid":
        print("WARNING: Invalid mask channel index")

    for indd in range(nd):
        sec = get_data_store_entry(config, indd)
        chan_str = sec.get('Channel', None)

        if(not isnone(chan_str) and (chan_str.find("AI") != -1
                                     or chan_str.find("CNT") != -1)):
            # Channel names are strings
            chan = chan_str.strip('"')
        else:
            # Channel names are integers
            chan = int(chan_str.strip('"'))

        if chan == chan_sel_str:
            mask[indm] = indd
            indm = indm + 1

    mask = mask[:indm]

    if isnone(mask):
        print("WARNING: Mask is empty")
    return mask


def get_idx_ts(config, indd):
    """Get the timestamp string of a data store entry."""
    sec = get_data_store_entry(config, indd)
    return sec.get('Timestamp', None)


def get_idx_ts_ms(config, mask):
    """Get the timestamps of the index entries in ms.

    The values are returned relative to the first element in the index.
    """
    ts_ofs = 0
    ts = np.ndarray(mask.size)

    for ind in range(mask.size + 1):
        if ind == 0:
            indd = 0
        else:
            indd = mask[ind-1]

        ts_str = get_idx_ts(config, indd)
        s = ts_str.split(' ')[1]
        [s_chunk, ms_chunk] = s.split('.')
        ms_val = int(ms_chunk[:-1])
        [h_chunk, m_chunk, s_chunk] = s.split(':')
        s_val = int(s_chunk)
        m_val = int(m_chunk)
        h_val = int(h_chunk)
        ts_ms = ms_val + (s_val + (m_val + h_val*60)*60)*1000

        if ind == 0:
            ts_ofs = ts_ms
        else:
            ts[ind-1] = ts_ms - ts_ofs

    return ts


def get_data_item_z_pos(config, idx):
    """Get the Z position in mm of a data item at the given index."""
    sec = get_data_store_entry(config, idx)
    return float(sec.get('Z', None))


def get_idx_z_pos(config, mask=None):
    """Get the Z positions of the masked data store entries in mm."""
    if mask is None:
        mask = np.arange(0, get_data_store_idx_len(config))

    Z = np.ndarray(mask.size)
    for ind in range(mask.size):
        indd = mask[ind]
        Z[ind] = get_data_item_z_pos(config, indd)

    return Z


def get_frame_z_pos_arr(config):
    """Get frame Z positions in mm."""
    num_f = get_num_frames(config)
    num_ch = get_num_chan(config)

    z_arr = np.ndarray(num_f)
    for ind_f in range(num_f):
        idx = ind_f*num_ch
        z_arr[ind_f] = get_data_item_z_pos(config, idx)

    return z_arr


def get_z_pos_rng(config):
    """Get the range of frame Z positions in mm."""
    z_arr = get_frame_z_pos_arr(config)
    return [np.min(z_arr), np.max(z_arr)]


def get_z_pos_spa(config):
    """Get the span of Z positions in mm."""
    z_rng = get_z_pos_rng(config)
    return z_rng[1] - z_rng[0]


def get_z_pos_avg_step(config):
    """Get the average Z step between frames in mm."""
    z_arr = get_frame_z_pos_arr(config)
    return np.mean(np.diff(z_arr))


def get_cfg_range(config, chan_id=2):
    """Get the display mapping range for a given channel from the config."""
    rng = get_head_val(config, "Channel " + str(chan_id), "Range")
    if rng == '' or rng is None:
        return None

    rng.strip('"')
    rng = rng.split(',')
    return [int(rng[0]), int(rng[1])]


def get_cfg_gamma(config, ch=2):
    """Get the config gamma value for a given channel."""
    gamma = get_head_val(config, "Channel " + str(ch), "Gamma")
    if(gamma == '' or gamma is None):
        return None
    return float(gamma.strip('"'))


def get_data_type(config=None, file_name=None):
    """Determine data type from the config file."""
    if isnone(config):
        config = read_cfg(file_name)

    scan_type = get_head_val(config, "Scan Info", "Type")
    if scan_type is None:
        scan_type = get_head_val(config, "Scan", "Type")

    if scan_type is not None:
        scan_type = scan_type.strip(r'"')

    if scan_type.lower().find("tiling scan") != -1:
        return DataType.Tiling
    if scan_type.lower().find("average") != -1:
        return DataType.Average
    else:
        num = get_data_store_idx_len(config)

        # TODO: This needs to be fixed for old data # pylint: disable=W0511
        # If there are only three or four channels the data must be a single
        # scan
        if num in (3, 4):
            return DataType.SingleImage

        # Get Z positions
        Z = get_idx_z_pos(config)

        if np.std(Z) == 0:
            # If the Z values are all the same the data must be a time lapse
            return DataType.TimeLapse
        else:
            # If the Z values are different the data must be a Z stack
            return DataType.ZStack

    # If somehow none of the data type guesses fit mark the type as invalid
    return DataType.Invalid


def print_data_info(config=None):
    """Print information about the dataset."""
    dtype = get_data_type(config=config)
    print("Data type: " + get_data_type_str(dtype))

    num_ch = get_num_chan(config)
    print("Number of channels: " + str(num_ch))

    print("Channels: ", end='')
    for ch_ind in range(num_ch):
        if ch_ind < num_ch-1:
            print(get_chan_name(config, ch_ind) + ', ', end='')
        else:
            print(get_chan_name(config, ch_ind))

    print("Number of frames: " + str(get_num_frames(config)))

    pixel_t = get_px_time(config)
    rep_f = get_ex_rep_rate(config)
    print("Pixel dwell time: {:2g} us".format(pixel_t*1E6))
    print("Laser rep. rate: {:2g} MHz".format(rep_f*1E-6))

    limits = get_px_cnt_limits(config)
    print("Maximum pixel count limits:")
    print("\tRep. rate: {:d} c.".format(limits.RepRate))
    print("\tSingle-pulse: {:d} c.".format(limits.SinglePulse))
    print("\tCount linearity, 10% loss: {:d} c.".format(limits.CountLinearity))

    frame_t = get_scan_frame_time(config)
    scan_t = get_total_scan_time(config)
    meas_t = get_total_meas_time(config)
    overhead_t = meas_t - scan_t
    print("Frame scan time: {:.3g} s".format(frame_t))
    print("Total scan time: " + make_human_time_str(scan_t))
    print("Measurement time: " + make_human_time_str(meas_t))
    print("Scan overhead: " + make_human_time_str(overhead_t))
    print("Measurement scan time " +
          "efficiency: {:.3g}".format(1-overhead_t/meas_t))
    print("\n")

    if dtype == DataType.ZStack:
        z_rng = get_z_pos_rng(config)
        z_step = get_z_pos_avg_step(config)
        z_span = get_z_pos_spa(config)
        print("Z stack scan config:")
        print("\tFrom: {:.3g} um".format(z_rng[0]*1E3))
        print("\tTo: {:.3g} um".format(z_rng[1]*1E3))
        print("\tSpan: {:3g} um".format(z_span*1E3))
        print("\tAvg step: {:.3g} um".format(z_step*1E3))


def get_tiling_cfg(config):
    """Get the tiling configuration.

    The configuration is returned as a [FromX, ToX, FromY, ToY, Step] array.
    """
    sec_str = "Tiling Config"
    sec = get_cfg_section(config, sec_str)
    if isnone(sec):
        print("No tiling configuration in header")
        return None

    from_x = get_head_val(config, sec_str, "From X", "float", 0)
    to_x = get_head_val(config, sec_str, "To X", "float", 0)
    from_y = get_head_val(config, sec_str, "From Y", "float", 0)
    to_y = get_head_val(config, sec_str, "To Y", "float", 0)
    step = get_head_val(config, sec_str, "Step", "float", 0)

    return [from_x, to_x, from_y, to_y, step]


def get_tiling_step(config):
    """Get the tiling step size."""
    sec_str = "Tiling Config"
    sec = get_cfg_section(config, sec_str)

    if isnone(sec):
        print("No tiling configuration in header")
        return None

    return get_head_val(config, sec_str, "Step", "float", 0)
