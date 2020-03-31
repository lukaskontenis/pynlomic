
"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains routines to process microscopy data and images.

Some ideas are taken from my collection of Matlab scripts developed while in
the Barzda lab at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from enum import Enum
import matplotlib.image as mpimg

import os
import re
import sys
from shutil import copyfile

from lklib.util import isarray, isnone, issmth, reduce_trace, isstring, \
    handle_general_exception, get_color
from lklib.string import make_human_time_str, timestamp_str_to_seconds    
from lklib.fileread import ReadBigFile, ReadBinaryDataFile, ReadStarLabFile, \
    ReadTDMSTraces, ReadLCBeamDiagnostics, GetLCBeamDiagnosticsPath, \
    ListFilesWithExtension, ListFilesWithFilter, RemoveFileExtension, \
    ChangeExtension
from lklib.cfgparse import ReadConfig, GetHeaderValue, GetConfigSection
from lklib.image import crop_rem_rrcc, remap_img, get_colourmap, \
    bake_cmap, gen_preview_img, corr_field_illum, \
    add_scale_bar, show_img, normalize, tile_img, tile_img_blind, \
    get_frac_sat_rng, save_img, get_img_sz
from lklib.plot import export_figure, plot_trace, show_png_ext
from lklib.trace import Trace, trace_set_param, read_tdms_trace_ophir, get_pharos_log_trace
from lklib.report import MakeSVGReport, ConvertSVGToPDF


class ImageStats:
    Type = None
        
class VoltageImageStats(ImageStats):
    MinLevel = None
    AvgLevel = None
    MaxLevel = None
    
class CountImageStats(ImageStats):
    TotalCount = None
    MaxCount = None

class DetectorType(Enum):
    Counter=1
    Voltage=2

class MosaicType( Enum ):
    TimeSeries = 1
    ZStack = 2
    

class DataType( Enum ):
    """
    Data file type enum.
    """
    Invalid = 0
    SingleImage = 1
    Average = 2
    TimeLapse = 3
    ZStack = 4
    Tiling = 5

def GetSampleName( config ):
    """
    Get the name of the sample.
    """
    
    if( isarray( config )):
        return GetSampleName( config[0] )
    
    return GetHeaderValue( config, "Sample", "Name" )


def GetSampleID( config ):
    """
    Get the ID of the sample.
    """
    return GetHeaderValue( config, "Sample", "ID" )


def GetSampleAreaLabel( config ):
    """
    Get the area label of the sample.
    """
    return GetHeaderValue( config, "Sample", "Area" )


def GetScanDate( config ):
    """
    Get the scan date.
    """
    return GetHeaderValue( config, "Scan Info", "Scan date" )


def GetScanDescription( config ):
    """
    Get the scan description string.
    """
    return GetHeaderValue( config, "Scan Info", "Description" )


def GetOperatorName( config ):
    """
    Get the operator name.
    """
    return GetHeaderValue( config, "Scan Info", "Operator" )    

    
def GetNumChan(config):
    """
    Get the number of channels in dataset.
    """
    
    for chan_id in range(255):
        if isnone(GetConfigSection(config, "Channel " + str(chan_id))):
            return chan_id
        
    raise Exception("InvalidChannelNumber")
    
    return None    


def GetChannelName( config, chan ):
    """
    Get the name of channel 'chan'.
    """
    
    if( isarray( config )):
        numD = len( config )
        names = []
        
        for indD in range( 0, numD ):
            names.append( GetChannelName( config[indD], chan ) )
            
        return names

    return GetHeaderValue( config, "Channel " + str( chan ), "Name" )


def GetChannelHWName(config, chan_ind):
    """
    Get the hardware name of the given chanel index. This information should be
    added to the config file.
    """
    hw_chan_map = {
        0: "AI0",
        1: "AI1",
        2: "CNT0",
        3: "CNT1"
    }
    return hw_chan_map.get(chan_ind, "invalid")

def GetChannelDetectorType(config, chan_ind):
    """
    Get the detector type of the given channel index.
    TODO: This should read the config file to determine the detector.
    """
    ValidateChannelIndex(config, chan_ind)
    if(chan_ind==0 or chan_ind==1):
        chan_type = DetectorType.Voltage
    elif(chan_ind==2 or chan_ind==3):
        chan_type = DetectorType.Counter
    return chan_type


def GetChannelUnits(chan_type):
    """
    Get the units of a channel based on its type.
    """
    if(chan_type == DetectorType.Voltage):
        chan_units = "V"
    elif(chan_type == DetectorType.Counter):
        chan_units = "c."
    return chan_units

def GetDefaultChannelIndex(config):
    """
    Get the default channel index when one is not explicitly provided.
    """
    return 2

def ValidateChannelIndex(config, chan_ind):
    """
    Validate channel index.
    TODO: This should read the config file to check which channels are defined.
    """
    if(chan_ind<0 or chan_ind>2):
        print("Unsupported channel index")
        return False
    else:
        return True
    
    
    
def GetNLOrd( config, chan ):
    """
    Get the nonlinear order of channel 'chan'.
    """
    
    name = GetChannelName( config, chan )
    
    if( name == "SHG" ):
        return 2
    elif( name == "THG" ):
        return 3
    elif( name[0:2] == "Ax" ):
        # Alexa MPEF
        return 2    
    else:
        print( "Unknown nonlinear order of channel '" + str(chan) + "', assuming 2" )
        return 2
    
    return -1


def GetChannelFilterName( config, chan ):
    """
    Get the name of the filter for a given channel.
    """
    
    name = GetHeaderValue( config, "Channel " + str( chan ), "Filter" )
    
    if( isnone( name )):
        return None
    else:
        return name.strip( r'"' )


def GetLaserName( config ):
    """
    Get laser name.
    """
    return GetHeaderValue( config, "Setup", "Laser" )


def GetExWavelength( config ):
    """
    Get the excitation wavelength in um.
    """
    if( isarray( config )):
        return GetExWavelength( config[0] )
    
    exw_str = GetHeaderValue( config, "Setup", "Laser wavelength" )
    
    if( exw_str != None ):
        try:
            exw = float( exw_str )
        except ValueError:
            print( "Cannot parse laser wavelength" )
            exw = None
            
        return exw
    else:
        return None
    
def GetExPower(config):
    """
    Get the excitation power in W.
    """
    if( isarray( config )):
        return GetExPower(config[0])
    
    exP_str = GetHeaderValue( config, "Setup", "Laser power" )
    
    if( exP_str != None ):
        try:
            exP = float( exP_str )
        except ValueError:
            print( "Cannot parse laser power" )
            exP = None
            
        return exP
    else:
        return None

    
def GetDetSatThreshold(config):
    """
    Get the count linearity of the detector.
    """
    # Assuming H10682 photon counter
    return 5E6

class PixelCountLimit:
    RepRate = None
    SinglePulse = None
    CountLinearity = None

def GetPixelCountLimits(config):
    """
    Get the maximum expected number of counts per pixel.
    """
    
    pixel_t = GetPixelTime(config)
    rep_f = GetExRepRate(config)
    sat_f = GetDetSatThreshold(config)
    
    limit = PixelCountLimit()
    limit.RepRate = int(np.round(pixel_t*rep_f))
    limit.SinglePulse = int(np.round(pixel_t*rep_f*0.1))
    limit.CountLinearity = int(np.round(pixel_t*sat_f))
    return limit

def GetPixelCountLimit(config,ignore_linearity_limit = True):
    """
    Get the maximum expected number of counts per pixel.
    """
    limit = GetPixelCountLimits(config)
    
    if(ignore_linearity_limit):
        return np.min([limit.RepRate, limit.SinglePulse])
    else:
        return np.min([limit.RepRate, limit.SinglePulse, CountLinearity])
    
def GetPixelBackgroundCount(config):
    """
    Get the pixel count expected due to background. Using photon counters
    at us dwell times results in at most 1 count per pixel due to dark counts.
    This then assumes a background level of 2 c./px
    """
    return 2

def GetStagePosition( config, axis = None, index = None ):
    """
    Get the stage position of a given axis. If index is specified its position
    is returned, otherwise the reference stage position is returned.
    """
    
    if( isnone( index )):
        pos_str = GetHeaderValue( config, "Position", axis )
    else:
        pos_str = GetHeaderValue( config, "Index " + str( index ), axis )
    
    pos = None
    if( pos_str != None ):
        try:
            pos = float( pos_str )
        except ValueError:
            print( "Cannot parse stage position" )
            
    return pos
    
    
def GetStageXYZPosition( config = None, FileName = None, index = None ):
    
    if( isarray( FileName )):
        FileNames = FileName
        
        pos = np.ndarray( [ len(FileNames), 3 ])
        for ( ind, FileName ) in enumerate( FileNames ):
            pos[ ind, : ] = GetStageXYZPosition( FileName = FileName )
            
        return pos
    
    if( config == None and FileName == None ):
        print( "Config object or file name must be supplied" )
        
    if( isnone( config )):
        config = ReadConfig( FileName )
    
    X = GetStagePosition( config, "X", index = index )
    Y = GetStagePosition( config, "Y", index = index )
    Z = GetStagePosition( config, "Z", index = index )
    
    return [ X, Y, Z ]

def GetExRepRate( config ):
    """
    Get the repetition rate of the excitation source.
    """
    
    laser = GetLaserName( config )
    
    if(isnone(laser)):
        return None
    
    laser = laser.strip( r'"' )
    
    if( laser == "FLINT" ):
        return 76E6
    
    if( laser == None ):
        print( "No laser specified" )
        return None
    
    s = "Cannot determine repetition rate for " + laser
    print( s )
    return None        
    
    
def GetScanFieldSize( config, apply_sz_calib = True ):
    """
    Get the scan field size in um from the config file.
    """
    
    fieldsz_um = GetHeaderValue( config, "Scan Geometry", "Field size" )
    if( fieldsz_um == None ):
        fieldsz_um = GetHeaderValue( config, "Scan", "Field size" )
    
    if( fieldsz_um == None ):
        print( "Cannot find scan field size in header" )
        return None
    
    fieldsz_um = float( fieldsz_um )
    
    if( apply_sz_calib ):
        calib_corr = GetScanFieldCalibCorrection( config )
        fieldsz_um = fieldsz_um*calib_corr
        
    return fieldsz_um


def GetScanFrameTime( config ):
    """
    Get the scan frame time in seconds from the config file.
    """
    
    frameT_s = GetHeaderValue( config, "Scan Geometry", "Frame time")
    if( frameT_s == None ):
        frameT_s = GetHeaderValue( config, "Scan", "Frame time")
        
    if( frameT_s == None ):
        print( "Cannot find frame time in header" )
        return None
    
    return float( frameT_s )


def GetPixelTime( config ):
    """
    Get pixel dwell time in seconds.
    """
    
    frameT_s = GetScanFrameTime( config )
    res = GetScanResolution( config )
    
    if( frameT_s == None or res == None ):
        print( "Cannot determine pixel time" )
        return None
    
    return frameT_s/res/res


def GetTotalScanTime(config):
    """
    Get the total scan time in seconds. Scan time excludes stage movement and
    other overhead.
    """
    num_f = GetNumFrames(config)
    frame_t = GetScanFrameTime(config)
    
    return num_f * frame_t


def GetTotalMeasurementTime(config):
    """
    Get the total measurement time in seconds.
    """
    num_idx = GetDataStoreIndexLength(config)
    return timestamp_str_to_seconds(GetIndexTimestamp(config,num_idx-1)) - \
        timestamp_str_to_seconds(GetIndexTimestamp(config,0))


def GetScanResolution( config ):
    """
    Get the scan resolution in pixels from the config file.
    """
    
    numR = GetHeaderValue( config, "Scan Geometry", "Lines" )
    numC = GetHeaderValue( config, "Scan Geometry", "Columns" )
    
    if( numR == None or numC == None ):
        numR = GetHeaderValue( config, "Scan", "Lines" )
        numC = GetHeaderValue( config, "Scan", "Columns" )
        
    if( numR == None or numC == None ):
        print( "Cannot find scan resolution in header" )
        return None
    
    numR = float( numR )
    numC = float( numC )
    
    if numR != numC:
        print( 'Image is not square!' )
        
    return ( numR + numC )/2


def GetScanFieldCalibCorrection( config ):
    """
    Get the calibration correction value for the physical scan field size. This
    function should be used to correct previous scan data if the scan field
    calibration is later determined to be wrong.
    """
    
    scan_field_calib_date = None
    has_calib = False
    if 'Calibration' in config.sections():
        has_calib = True
        calibcfg = config['Calibration']
        scan_field_calib_date = calibcfg.get('Scan field calib date',None)
    
    # IF no scan field calibration is defined or it is older than 2018.04.04, the
    # scan field size calibration is wrong and needs to be corrected
    if ( has_calib == False  ) or ( has_calib == True and ( scan_field_calib_date < datetime(2018, 4, 4) )):
        return 0.785
    
    # Otherwise all is fine
    return 1.0


def GetScanPixelSize( config, **kwargs ):
    """
    Get the size of the scan pixel in um/px from the config file.
    """
    
    if( isarray( config )):
        numD = len(config)
        umpx = np.ndarray( numD, float )
        for indD in range( 0, numD ):
            umpx[indD] = GetScanPixelSize( config[indD] )
            
        if( (umpx.mean() == umpx).all() == False ):
            print( "Unequal pixel sizes" )
            
        return umpx.mean()        
            
    field_sz_um = GetScanFieldSize( config, **kwargs )
    img_res_col = GetScanResolution( config )
    if( field_sz_um == None or img_res_col == None ):
        print( "Cannot determine pixel size" )
        return None
    else:
        umpx = field_sz_um/img_res_col
        return umpx


def GetDataStoreIndexLength( config ):
    """
    Get the length of the data store index.
    """
    
    num_sec = 0
    while(1):
        sec_name = 'Index ' + str(num_sec)
        if sec_name not in config.sections():
            return num_sec
        num_sec = num_sec + 1
        
        
def GetDataStoreIndexEntry( config, ind ):
    """
    Get the specified data index entry.
    """
    
    sec_name = 'Index ' + str(ind)
    return config[ sec_name ]


def GetNumFrames(config):
    """
    Get the number of frames.
    """
    num_idx = GetDataStoreIndexLength(config)
    num_ch = GetNumChan(config)
    
    num_f = num_idx/num_ch 
    if(num_f % 1 != 0):
        print("WARNING: Number of frames in not a round number")
        
    return int(num_f)


def GetIndexMask( config, chan_sel ):
    """
    Get the data index mask for the specified channel.
    """
    
    numD = GetDataStoreIndexLength( config )
    indM = 0
    mask = np.ndarray( numD, dtype=int )
    
    
    chan_sel_str = GetChannelHWName(config, chan_sel)
    if(chan_sel_str == "invalid"):
        print("WARNING: Invalid mask channel index")
    
    for indD in range( numD ):
        sec = GetDataStoreIndexEntry( config, indD )
        chan_str = sec.get('Channel',None)
        
        if( not isnone( chan_str ) and ( chan_str.find( "AI" ) != -1 or chan_str.find( "CNT" ) != -1 )):
            # Channel names are strings
            chan = chan_str.strip('"')
        else:
            # Channel names are integers
            chan = int( chan_str.strip('"') )
            
        if( chan == chan_sel_str ):
            mask[indM] = indD
            indM = indM + 1
            
    mask = mask[ :indM ]
    
    if(isnone(mask)):
        print("WARNING: Mask is empty")
    return mask


def GetIndexTimestamp( config, indD ):
    """
    Get the timestamp string of the specified index entry.
    """
    
    sec = GetDataStoreIndexEntry( config, indD )
    return sec.get('Timestamp',None)


def GetIndexTimestamps_ms( config, mask ):
    """
    Get the timestamps of the index entries in milliseconds relative to the
    first element in the index.
    """
    
    ts_ofs = 0
    ts = np.ndarray( mask.size )
    
    for ind in range( mask.size + 1 ):
        if( ind == 0 ):
            indD = 0
        else:
            indD = mask[ind-1]
        
        ts_str = GetIndexTimestamp( config, indD )
        
        S = ts_str.split( ' ' )[1]
        
        [ S, ms ] = S.split('.')
        
        ms = int(ms[:-1])
        
        [ h, m, s ] = S.split(':')
        
        s = int(s)
        m = int(m)
        h = int(h)
        
        ts_ms = ms + (s + ( m+h*60)*60 )*1000
        
        if( ind == 0 ):
            ts_ofs = ts_ms
        else:
            ts[ ind-1 ] = ts_ms - ts_ofs
            
    return ts

def GetDataItemZPos(config, idx):
    """
    Get the Z position of a data item at the given index.
    Units: mm
    """
    sec = GetDataStoreIndexEntry(config, idx)
    return float(sec.get('Z',None))
   
def GetIndexZPos(config, indD):
    """
    Get the Z position of an entry in millimeters.
    """
    print("DEPRECATED: GetIndexZPos is deprecated")
    return GetDataItemZPos(config, indD)
    


def GetIndexZPositions(config, mask = None):
    """
    Get the Z positions of the masked entries in millimetres.
    """
    
    if( mask == None ):
        mask = np.arange( 0, GetDataStoreIndexLength( config ))
        
    Z = np.ndarray( mask.size )
    
    for ind in range( mask.size ):
        indD = mask[ind]
        Z[ ind ] = GetDataItemZPos( config, indD )
            
    return Z

def GetFrameZPosArray(config):
    """
    Get frame Z positions.
    Units: mm.
    """
    num_f = GetNumFrames(config)
    num_ch = GetNumChan(config)
    
    z_arr = np.ndarray(num_f)
    
    for ind_f in range(num_f):
        idx = ind_f*num_ch
        z_arr[ind_f] = GetDataItemZPos(config,idx)
        
    return z_arr

def GetZPosRange(config):
    """
    Get the range of frame Z positions.
    Units: mm.
    """
    z_arr = GetFrameZPosArray(config)
    return [np.min(z_arr), np.max(z_arr)]

def GetZPosSpan(config):
    """
    Get the span of Z positions.
    Units: mm.
    """
    z_rng = GetZPosRange(config)
    return z_rng[1] - z_rng[0]

def GetZPosAvgStep(config):
    """
    Get the average Z step between frames.
    Units: mm
    """
    z_arr = GetFrameZPosArray(config)
    return np.mean(np.diff(z_arr))

def GetConfigRange( config, chan_id = 2 ):
    """
    Get the config display mapping range for a given channel.
    """
    
    rng = GetHeaderValue( config, "Channel " + str( chan_id ), "Range" )
    
    if( rng == '' or rng == None ):
        return None
    
    rng.strip('"')
    rng = rng.split( ',' )
    
    return [ int(rng[0]), int(rng[1])]


def GetConfigGamma( config, ch = 2 ):
    """
    Get the config gamma value for a given channel.
    """
    
    gamma = GetHeaderValue( config, "Channel " + str(ch), "Gamma" )
    
    if( gamma == '' or gamma == None ):
        return None
    
    gamma.strip('"')
    
    return float(gamma)


def GetDataType( config = None, FileName = None ):
    """
    Determine data type from the config file.
    """
    
    if( isnone( config )):
        config = ReadConfig( FileName )
    
    t = GetHeaderValue( config, "Scan Info", "Type" )
    if( t == None ):
        t = GetHeaderValue( config, "Scan", "Type" )
    
    if t != None:
        t = t.strip( r'"' )
    
    if t.lower().find( "tiling scan" ) != -1:
        return DataType.Tiling
    if t.lower().find( "average" ) != -1:
        return DataType.Average
    else:
        num = GetDataStoreIndexLength( config )
    
        # TODO: This needs to be fixed for old data
        # If there is only three or fourchannels the data must be a single scan    
        if( num == 3 or num == 4 ):
            return DataType.SingleImage
        
        # Get Z positions
        Z = GetIndexZPositions( config )
        
        if( np.std( Z ) == 0 ):
            # If the Z values are all the same the data must be a time lapse
            return DataType.TimeLapse
        else:
            # If the Z values are different the data must be a Z stack
            return DataType.ZStack
        
    # If somehow none of the data type guesses fit mark the type as invalid
    return DataType.Invalid


def PrintDataInfo(config = None):
    """
    Print information about the dataset.
    """
    dtype = GetDataType(config = config)
    
    print("Data type: " + GetDataTypeString(dtype))
    
    num_ch = GetNumChan(config)
    print("Number of channels: " + str(num_ch))
    
    print("Channels: ", end='')
    for ch_ind in range(num_ch):
        if(ch_ind < num_ch-1):
            print(GetChannelName(config, ch_ind) + ', ', end='')
        else:
            print(GetChannelName(config, ch_ind))
            
    print("Number of frames: " + str(GetNumFrames(config)))
    
    pixel_t = GetPixelTime(config)
    rep_f = GetExRepRate(config)
        
    print("Pixel dwell time: {:2g} us".format(pixel_t*1E6))
    print("Laser rep. rate: {:2g} MHz".format(rep_f*1E-6))
    
    limits = GetPixelCountLimits(config)
        
    print("Maximum pixel count limits:")
    print("\tRep. rate: {:d} c.".format(limits.RepRate))
    print("\tSingle-pulse: {:d} c.".format(limits.SinglePulse))
    print("\tCount linearity, 10% loss: {:d} c.".format(limits.CountLinearity))
    
    frame_t = GetScanFrameTime(config)
    scan_t = GetTotalScanTime(config)
    meas_t = GetTotalMeasurementTime(config)
    overhead_t = meas_t - scan_t
        
    print("Frame scan time: {:.3g} s".format(frame_t))
    print("Total scan time: " + make_human_time_str(scan_t))
    print("Measurement time: " + make_human_time_str(meas_t))
    print("Scan overhead: " + make_human_time_str(overhead_t))
    print("Measurement scan time efficiency: {:.3g}".format(1-overhead_t/meas_t))
    print("\n")    
    
    if(dtype == DataType.ZStack):
        z_rng = GetZPosRange(config)
        z_step = GetZPosAvgStep(config)
        z_span = GetZPosSpan(config)
        print("Z stack scan config:")
        print("\tFrom: {:.3g} um".format(z_rng[0]*1E3))
        print("\tTo: {:.3g} um".format(z_rng[1]*1E3))
        print("\tSpan: {:3g} um".format(z_span*1E3))
        print("\tAvg step: {:.3g} um".format(z_step*1E3))
        

def GetDataTypeString(DType):
    """
    Return the name of the data type as a string.
    """
    if(DType == DataType.SingleImage):
        return "Singe image"
    
    if(DType == DataType.Average):
        return "Average"
    
    if(DType == DataType.TimeLapse):
        return "Time lapse"
    
    if(DType == DataType.Tiling):
        return "Tiling"
    
    if(DType == DataType.ZStack):
        return "Z Stack"
    
    return "INVALID DATA TYPE"    

def GetTilingConfig( config ):
    """
    Get the tiling configuration as [ FromX, ToX, FromY, ToY, Step ].
    """
    
    sec_str = "Tiling Config" 
    sec = GetConfigSection( config, sec_str )
    
    if( isnone( sec )):
        print( "No tiling configuration in header" )
        return None
    
    from_X = GetHeaderValue( config, sec_str, "From X", "float", 0 )
    to_X = GetHeaderValue( config, sec_str, "To X", "float", 0 )
    from_Y = GetHeaderValue( config, sec_str, "From Y", "float", 0 )
    to_Y = GetHeaderValue( config, sec_str, "To Y", "float", 0 )
    step = GetHeaderValue( config, sec_str, "Step", "float", 0 )
    
    return [ from_X, to_X, from_Y, to_Y, step ]

def GetTilingStep( config ):
    """
    Get the tiling step size.
    """
    
    sec_str = "Tiling Config" 
    sec = GetConfigSection( config, sec_str )
    
    if( isnone( sec )):
        print( "No tiling configuration in header" )
        return None
    
    return GetHeaderValue( config, sec_str, "Step", "float", 0 )
    

def GetChannelFrames( Data, config, chan = 2 ):
    """
    Get frames from the specified channel.
    """
    
    mask = GetIndexMask( config, chan )
    
    return Data[:,:,mask]


def GetChannelSum( Data, config, chan=2 ):
    """
    Sum the counts in the frames of the given channels.
    """
    
    F = GetChannelFrames( Data, config, chan=chan )
    
    return F.sum( 2 )


def GetDefaultColourmap( chan_str = None ):
    """
    Get default colourmap based on channel description.
    """
    
    if( chan_str == None ):
        return "magma"
    
    if( chan_str.find( "BP340" ) != -1 ):
        return "KBW_Nice"
    
    if( chan_str.find( "BP520" ) != -1 or chan_str.find( "BP550" ) != -1 ):
        return "KGW_Nice"
    
    if( chan_str.find( "BP600" ) != -1 ):
        return "KOW_Nice"
    
    if( chan_str.find( "BP650" ) != -1 ):
        return "KRW_Nice"
    
    return "magma"


def MakeImageTitle( config, template = "fig",
                   chan = None, print_exw = False, chas = None ):
    """
    Make an image title string.
    """
    
    chan_name_str = None
    
    sample_name = GetSampleName( config )
    if( not isnone( chan )):
        chan_name_str = GetChannelName( config, chan )
    
    laser_name = GetLaserName(config)
    if(not isnone(laser_name)):
        exw = GetExWavelength(config)
        exP = GetExPower(config)
    
    if( template == "fig" ):
        title_str = sample_name
                
        if(print_exw and not isnone(laser_name)):
            if(not isnone(exw) or not isnone(exP)):
                title_str = title_str + ", Ex. "
            if(not isnone(exw)):
                title_str = title_str + "%.2f um" %(exw)
            if(not isnone(exP)):
                title_str = title_str + "%.1f mW" %(exP)
        if( chan != None ):
            
            if( isarray( chan_name_str )):
                chan_pre = [ "R: ", "G: ", "B: " ]
                
                str2 = ''
                for ind in range( 0, len( chan_name_str )):
                    if( issmth( chas )):
                        ch_ind = chas[ind]
                    else:
                        ch_ind = ind
                        
                    str2 = str2 + chan_pre[ch_ind] + chan_name_str[ind] + '; '
                
                chan_name_str = str2
            
            if( chan_name_str != None ):
                title_str = title_str + ", " + chan_name_str
                
    elif( template == "report" ):
        title_str = sample_name + '\n'
        
        if( not isnone( chan_name_str )):
            title_str = title_str + chan_name_str
            
        if( not isnone( exw )):
            title_str = title_str + ', Ex. ' + str( exw ) + ' um'
            
    else:
        print( "Unsupported title template ''%s''" %template )
        title_str = None
        
    return title_str


def MakeCaptionString( config, template = "fig", ch_ind = None, rng = None, gamma = None, cm = None,
                      scalebar_sz = None, image_stats = None, img_sz = None ):
    """
    Make a caption string for the figure.
    """
    
    if(isnone(ch_ind)):
        ch_ind = GetDefaultChannelIndex
    
    ch_type = GetChannelDetectorType(config, ch_ind)

    if( isarray( config )):
        caption_str = ''
        numD = len( config )
        chan_pre = [ "R: ", "G: ", "B: " ]
        for indD in range( 0, numD ):
            caption_str = caption_str + chan_pre[indD] + \
                MakeCaptionString( config[indD], rng = rng[indD], gamma = gamma[indD] ) + '\n'
                
        caption_str = caption_str + 'bar = ' + str( scalebar_sz ) + " um "
                
        return caption_str
    
    caption_str = ''
    
    caption_str = caption_str + "Ch: %d" %(ch_ind)
    
    if( rng != None ):
        if(ch_type==DetectorType.Counter):
            rng_str = "[%d, %d]" %(rng[0], rng[1])
        elif(ch_type==DetectorType.Voltage):
            rng_str = "[%.1f, %.1f]" %(rng[0], rng[1])
        caption_str = caption_str + \
        ", range: %s %s" %(rng_str, GetChannelUnits(ch_type))
        
    if( rng != None ):
        if( gamma == 1 ):
            caption_str = caption_str + ", gamma: 1"
        else:
            caption_str = caption_str + ", gamma = %1.1f" % (gamma)
            
    if( cm != None ):
        caption_str = caption_str + ", cmap: " + cm
    
    if( template == "fig" and scalebar_sz != None ):
        caption_str = caption_str + ", bar = " + str( scalebar_sz ) + " um"
    
    if(ch_type==DetectorType.Counter):
        if( image_stats.TotalCount != None ):
            frameT = GetScanFrameTime( config )
            caption_str = caption_str + "\nAvg: %.2f Mcps" %( image_stats.TotalCount/frameT/1E6 )
        if( image_stats.MaxCount != None ):
            pixelT = GetPixelTime( config )
            caption_str = caption_str + ", max = %.2f Mcps" %( image_stats.MaxCount/pixelT/1E6 )
            
    elif(ch_type==DetectorType.Voltage):
        if(not isnone(image_stats.MinLevel)):
            caption_str = caption_str + "\nMin: %.2f V" %(image_stats.MinLevel)
        if(not isnone(image_stats.AvgLevel)):
            caption_str = caption_str + ", avg: %.2f V" %(image_stats.AvgLevel)
        if(not isnone(image_stats.MaxLevel)):
            caption_str = caption_str + ", max: %.2f V" %(image_stats.MaxLevel)
    
    if( template == "report" ):
        caption_str = caption_str + '\n'
    
        caption_str = caption_str + 'Tiling: '
        
        field_sz = GetScanFieldSize( config, apply_sz_calib = False )
        if( not isnone( field_sz )):
            caption_str = caption_str + str( field_sz ) + ' um size'
            
        tiling_grid_sz = GetTilingGridSize( config = config )
        if( not isnone( tiling_grid_sz )):
            caption_str = caption_str + ', %dx%d grid' %( tiling_grid_sz[0], tiling_grid_sz[1] )
            
        tiling_step_sz = GetTilingStep( config )
        if( not isnone( tiling_step_sz )):
            caption_str = caption_str + ', %.1f mm step' %tiling_step_sz
            
        pixel_sz = GetScanPixelSize( config, apply_sz_calib = False )
        if( not isnone( pixel_sz )):
            caption_str = caption_str + ', pixel size: %.2f um' %pixel_sz
            
        scan_area_X = img_sz[1] * pixel_sz
        scan_area_Y = img_sz[0] * pixel_sz
        if( not isnone( scan_area_X ) and not isnone( scan_area_Y )):
            caption_str = caption_str + ', scan area: %.2fx%.2f mm' %( scan_area_X/1E3, scan_area_Y/1E3 )
            
        image_num_Mpx = img_sz[0]*img_sz[1]
        if( not isnone( image_num_Mpx )):
            caption_str = caption_str + ', %.1f Mpx' %(image_num_Mpx/1E6)
            
        caption_str = caption_str + '\n'
        
        date = GetScanDate( config )
        if( not isnone( date )):
            caption_str = caption_str + 'Data: ' + date
            
        operator = GetOperatorName( config )
        if( not isnone( operator )):
            caption_str = caption_str + ', Scanned by: ' + operator
        
        sample_id = GetSampleID( config )
        if( not isnone( sample_id )):
            caption_str = caption_str + ', Sample: ' + sample_id
            
        sample_area_label = GetSampleAreaLabel( config )
        if( not isnone( sample_area_label )):
            caption_str = caption_str + ', Area ' + sample_area_label
            
    return caption_str


def GetScanArtefactSize( FileName = None, config = None ):
    """
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
    
    if( isnone( config )):
        config = ReadConfig( FileName )
    
    field_sz_um = GetScanFieldSize( config )
    frame_T_s = GetScanFrameTime( config )
    umpx = GetScanPixelSize( config )
    
    if( field_sz_um == None or frame_T_s == None or umpx == None ):
        print( "Cannot determine scan artefact size" )
        return None
    
    # Observed artefact sizes for given scan field sizes and frame times
    S = [ 780, 393, 157, 78, 39]
    T = [ 10, 2.5, 0.8, 1, 1 ]
    cropL_um = [ 41.5, 27.5, 16.5, 3.14, 2.4 ]
    
    # Scan field size seems to be the largest factor. Find the closest
    # empirical scan field size for the current one
    ind = S.index( min( S, key=lambda x:abs(x-field_sz_um)))
    
    # Assume linear scaling with deviation from the empirical scan time for
    # the corresponding scan field size.
    T_fac = T[ind]/frame_T_s
    
    cropL_um1 = cropL_um[ind]
    
    # umpx seems to play a role as well. For a field size of 780 and pixel size
    # of 0.78 um the artefact is 42 um, but when pixel size is 0.39 um the
    # artefact becomes 91 um for some reason.
    if( ind == 0 and umpx < 0.31 ):
        cropL_um1 = 91
    else:
        cropL_um1 = cropL_um[ind]
        
    # Apply frame time scaling
    cropL_um1 = cropL_um1*T_fac
    
    # Convert crop size in um to px
    cropL_px = int( cropL_um1/umpx )
    
    return cropL_px


def CropScanArtefacts( img, config ):
    """
    crop the sides of the image corrupted by galvo scanning artefacts.
    """
    
    cropL_px = GetScanArtefactSize( config = config )
    
    img = crop_rem_rrcc( img, 0, 0, cropL_px, 0 )
    return img


def GetSatMask( img, config ):
    """
    Get a mask showing saturated pixels in image.
    """
    
    pxT = GetPixelTime( config )
    f = GetExRepRate( config )
    
    if( pxT == None or f == None ):
        print( "Cannot determine saturation level" )
        return None
    
    sat_level = f/10 * pxT
    
    mask = img / sat_level
    #mask[ mask < 1 ] = 0
    return mask


def GetDefaultChanColourmap( config, ch = 2 ):
    ch_name = GetChannelName( config, chan = ch )
    if( ch_name == "DAPI" ):
        return "KPW_Nice"
    elif( ch_name == "SHG" ):
        return "viridis"
    elif( ch_name == "THG" ):
        return "inferno"
    else:
        return GetDefaultColourmap( GetChannelFilterName( config, chan = ch ) )

def GetImage( **kwargs ):
    [ I, rng, gamma, D ] = ProcessImage( **kwargs )
    
    return I


def ProcessImage( FileName = None, rng = None, gamma = None, ch = 2, corr_fi = False ):
    """
    Process an image for analysis and display. Obtain specified mapping range
    and gamma values, crop scan artefacts and correct field illumination.
    """
    
    D = ReadBinaryDataFile( FileName )
    
    config = ReadConfig( FileName )
    
    if( rng == None ):
        rng = GetConfigRange( config, chan_id = ch )
        
    if( gamma == None ):
        gamma = GetConfigGamma( config, ch = ch )
    
    if( gamma == None ):
        gamma = 1
    
    D_type = GetDataType( config = config )
    if(  D_type == DataType.SingleImage or D_type == DataType.Average ):
        
        if( D_type == DataType.SingleImage ):
            I = D[ :, :, ch ]
        
        if( D_type == DataType.Average ):
            I = GetChannelSum( D, config, chan = ch )
            
        # Convert image to volts for analog channels
        # Assuming channel range is +-10V, no offset and 16bits
        if(ch == 0 or ch == 1):
            I = (I.astype('float')/2**16 - 0.5)*20
            
        I = CropScanArtefacts( I, config )
        
        if( corr_fi == True ):
            print( "Correcting field illumination..." )
            I = corr_field_illum( I, facpwr = GetNLOrd( config, ch ))
            
        if( isnone( rng )):
            rng = GetOptimalMappingRange( I = I, FileName = FileName )
            
    return [ I, rng, gamma, D ]


def MakeImage( I = None, D = None, FileName = None, rng = None, gamma = None, ch = 2, corr_fi = True, cm = None, cm_sat = False ):
    """
    Make an image for display.
    """
    if( isnone( I ) or isnone( D )):
        [ I, rng, gamma, D ] = ProcessImage(FileName = FileName, rng = rng, gamma = gamma, ch = ch, corr_fi = corr_fi )
    
    config = ReadConfig( FileName )
        
    D_type = GetDataType( config = config )
    if(  D_type == DataType.SingleImage or D_type == DataType.Average ):
        
        I_raw = I
        
        [ numR, numC ] = I.shape
        
        if( cm_sat ):
            map_cap = False
        else:
            map_cap = True
               
        [ I, rng ] = remap_img( I, rng = rng, gamma=gamma, cap = map_cap )
        
        I_scaled = I
        
        if( isnone( cm )):
            cm = GetDefaultChanColourmap(config)
        
        I = bake_cmap( I/255, cmap = cm, remap = False, cm_over_val = 'r', cm_under_val = 'b' )
    else:
        if D_type == DataType.TimeLapse:
            mos_type = MosaicType.TimeSeries
        elif D_type == DataType.ZStack:
            mos_type = MosaicType.ZStack
        else:
            print( "Unknown data type" + str( D_type ))
            #return None
    
        ShowMosaic( D, FileName, mos_type=mos_type )
    
    return [ I, I_raw, I_scaled, cm, rng, gamma ]


def MakeMosaicImage( D = None, mask = None, ij = None, pad = 0.02,
                    remap = True, rng = [ 0, 20 ] ):
    """
    Arrange images from array D into a mosaic with given coordinates and padding.
    """
    
    [ numR, numC, numD ] = D.shape
    
    pad_px = np.int32( np.round( max([numR, numC])*pad ) )
    
    num_grid_rows = ij[:,0].max() + 1
    num_grid_cols = ij[:,1].max() + 1
    
    mosaic_R = num_grid_rows*numR + (num_grid_rows-1)*pad_px
    mosaic_C = num_grid_cols*numC + (num_grid_cols-1)*pad_px
    
    if( remap ):
        M_dtype = np.float64
    else:
        M_dtype = D.dtype
        
    M = np.ndarray( [ mosaic_R, mosaic_C ], dtype = M_dtype )
    
    for ( ind, indD ) in enumerate( mask ):
        [ grid_row, grid_col ] = ij[ ind, : ]
        
        row_ofs = grid_row*( numR + pad_px )
        col_ofs = grid_col*( numC + pad_px )
        
        if( remap ):
            I = remap_img( D[:,:,indD], rng = rng )[0]
        else:
            I = D[:,:,indD]
            
        I = np.fliplr( I )
        
        M[ row_ofs : numR + row_ofs, col_ofs : numC + col_ofs ] = I

    return M


def ShowMosaicImage( **kwargs ):
    """
    Make and show a mosaic image.
    """    
    M = MakeMosaicImage( **kwargs )
    plt.imshow( M )
    plt.axis('off')
    
    return M

    
def MakeMosaicFigure( D = None, mask = None, ij = None, pad = 0.02, rng = [ 0, 20 ] ):
    """
    Make a mosaic figure of images in D arranged according to row and column
    indices in ij.
    This doesn't work well because of automatic figure scaling which results in
    different horizontal and vertical pixel spacing even though wspace and
    hspace are the same.
    """
    num_grid_rows = ij[:,0].max() + 1
    num_grid_cols = ij[:,1].max() + 1
    
    grid = plt.GridSpec( num_grid_rows, num_grid_cols, wspace=pad, hspace=pad)
    
    indT = 0
    for indR in range( 0, num_grid_rows ):
        for indC in range( 0, num_grid_cols ):
            ax = plt.subplot( grid[ ij[indT, 0], ij[indT,1] ] )
            ax.set_aspect('equal')
            #plt.imshow( I )
            I = remap_img( D[:,:,mask[indT]], rng = rng )[0]
            I = np.fliplr( I )
            plt.imshow( I )
            plt.axis('off')        
            indT = indT + 1


def MakeMosaic( D, FileName, mos_type='tseries', ar=16/9, index_mask=None, det_ch=2 ):
    """
    Make a mosaic of 2D images in a 3D array.
    """
    [ numR, numC, numD ] = D.shape
    
    pad = np.int32( np.round( max([numR, numC])*0.1 ) )
    
    if index_mask == None:
        config = ReadConfig( FileName )
        mask = GetIndexMask( config, det_ch )
    
    numD = mask.size
    
    numMC = np.int32( np.ceil( np.sqrt( ar*numD )) )
    numMR = np.int32( np.ceil( numD/numMC ) )
    
    M = np.ndarray( [numR*numMR + (numMR-1)*pad, numC*numMC + (numMC-1)*pad, 4], dtype='uint8' )
    M.fill(255)
    
    image_coords = np.ndarray( [ numD, 2 ])
    
    indMR = 0
    indMC = 0
    for indM in range( numD ):
        indD = mask[ indM ]
        I = gen_preview_img( D[:,:,indD] )
        
        M_fromR = indMR*( numR + pad )
        M_toR = M_fromR + numR
        M_fromC = indMC*( numC + pad )
        M_toC = M_fromC + numC
        M[ M_fromR : M_toR, M_fromC : M_toC, : ] = I*255
        
        image_coords[ indM, : ] = [ M_fromR, M_fromC ]
        
        indMC = indMC + 1
        if( indMC == numMC ):
            indMR = indMR + 1
            indMC = 0
        
    return [ M, image_coords ]


def ShowMosaic( D, FileName, mos_type=None, ar=16/9, index_mask=None, det_ch=2 ):
    """
    Show a mosaic of images.
    """
    
    config = ReadConfig( FileName )
    if index_mask == None:
        mask = GetIndexMask( config, det_ch )
    
    [ M, image_coords ] = MakeMosaic( D, FileName, mos_type, ar )
    
    numC = D.shape[1]
   
    plt.imshow( M )
    plt.axis( 'off' )
    
    if mos_type == None:
        mos_type = MosaicType.TimeSeries
    
    if mos_type == MosaicType.TimeSeries:
        lbl = GetIndexTimestamps_ms( config, mask )/1000
        label_str_pre = 't= '
        label_str_suf = ' s'
        
    elif mos_type == MosaicType.ZStack:
        lbl = GetIndexZPositions( config, mask )
        label_str_pre = 'z= '
        label_str_suf = ' mm'
        
    else:
        print( 'Unknown mosaic type ' + str( mos_type ) ) 
        
    for ind in range( image_coords.shape[0] ):
        cap_str = str(lbl[ind])
        if ind == 0:
            cap_str = label_str_pre + cap_str + label_str_suf
            
        plt.text( image_coords[ind,1]+numC/2, image_coords[ind,0]-7, cap_str, horizontalalignment='center' )


def GetTileStageXYZPositions( config = None, file_name = None, force_reconstruct = False, **kwargs ):
    """
    Get the XYZ sample stage positions for individual tiles in a tiled scan.
    """     
    if( isnone( config )):
        config = ReadConfig( file_name )
        
    mask = GetIndexMask( config, 2 )
    
    pos = np.ndarray( [ len(mask), 3 ]) 
    
    for ( ind, index ) in enumerate( mask ):
        pos[ ind, : ] = GetStageXYZPosition( config, index = index )
        
    if( force_reconstruct or ( pos == 0 ).all() ):
        print( "No tiling coordinates in index. Reconstructing from tiling config." )
        
        [ from_x, to_x, from_y, to_y, step ] = GetTilingConfig( config )
        num_x_tiles = int( np.ceil( np.abs( to_x - from_x) )/step ) + 1
    
        z_pos = GetStagePosition( config, "Z" )
        
        for ind in range( 0, len( mask )):
            pos[ind, 0 ] = from_x + np.mod( ind, num_x_tiles ) * step
            pos[ind, 1 ] = from_y + np.floor( ind/num_x_tiles ) * step
            pos[ind, 2 ] = z_pos
            
    return pos

def GetTileIJIndices( **kwargs ):
    """
    Get tile i,j indices from tile stage centre positions.
    """
    bad_ij = False
    pos = GetTileStageXYZPositions( **kwargs )
    
    step = pos[1,0] - pos[0,0]
    
    ij = np.ndarray( [pos.shape[0],2], dtype = np.int32 )
    
    # This is probably swapped axes since X is ind=0 in pos and row is ind=0
    # in ij. But since such a swap is required anyway, it all works out.
    for ind in range( 0, pos.shape[0] ):
        ij[ind,0] = int( np.round( (pos[ind,0] - pos[0,0])/step ))
        ij[ind,1] = int( np.round( (pos[ind,1] - pos[0,1])/step ))

    # Reverse ij order to correspond to physical axis orientation        
    ij[:,0] = ij[:,0].max() - ij[:,0]
    ij[:,1] = ij[:,1].max() - ij[:,1]
    
    # Verify that the IJ indices are positive
    if( ij < 0 ).any():
        bad_ij = True
        print( "Some ij indices are negative!" )
        
    # Verify that there are no duplicate ij indices
    bad_ind = 0
    for ind1 in range( 0, ij.shape[0] ):
        for ind2 in range( ind1+1, ij.shape[0] ):
            if( ( ij[ind1,:] == ij[ind2,:] ).all() ):
                bad_ind = bad_ind + 1
                
    if( bad_ind ):
        bad_ij = True
        print( "There are %d duplicate ij indices!" % bad_ind )
        
    if( bad_ij ):
        print( "ij indices don't make sense." )
        if( kwargs.get( 'force_return_ij' ) ):
            return ij
        else:
            return None
    else:
        return ij


def GetTilingGridSize( **kwargs ):
    """
    Get tiling grid size.
    """
    ij = GetTileIJIndices( **kwargs )
    
    return [ max( ij[:,0] ), max( ij[:,1] ) ]
            

def ShowRawTiledImage( FileName = None, D = None,
                       rng = None, save_images = True ):
    """
    Show tiled images arranged in a mosaic with correct tiling geometry but
    ignoring tiling step size and overlap.
    """
    try:
        if( isnone( D )):
            D = ReadBinaryDataFile( FileName )
            
        [ D, mask, ij ] = GetTilingData( D = D, FileName = FileName )
            
        M = ShowMosaicImage( D = D, mask = mask, ij = ij, rng = rng )
        
        if( save_images ):
            save_img( M, ImageName = RemoveFileExtension(FileName) + "RawTiled",
                      cmap = "viridis", bake_cmap = True )
    except:
        handle_general_exception( "Could not generate raw tiled image" )
            

def TileImages_SBS( FileNames = None, Path = None, sort_by_Y = False ):
    """
    Tile images side-by-side. The input files can be provided as a list of file
    names or a path can be given where all .dat files reside.
    """
    
    if( isnone( FileNames ) and isnone( Path )):
        print( "Either FileNames or Path should be provided" )
        return None
    
    if( isnone( FileNames )):
        FileNames = ListFilesWithExtension( Path = Path, ext = "dat" )


    if( sort_by_Y ):
        pos = GetStageXYZPosition( FileName = FileNames )
    
        # Get image sort order by Y position
        tile_inds = np.argsort( pos[:,1])
    
        FileNames_sorted = [FileNames[i] for i in tile_inds]
        
        FileNames = FileNames_sorted
        
    numI = len( FileNames )
   
    # Tile images
    I_comb = normalize( ProcessImage( FileNames[0] )[0].astype( np.uint8 ) )
    for ind in range( 1, numI ):
        I1 = normalize( ProcessImage( FileNames[ind] )[0].astype( np.uint8 ) )
        
        I_comb = tile_img( I_comb, I1 )
        
    return I_comb

def GetOptimalMappingRange( I = None, D = None, FileName = None, mask = None, ij = None ):
    if( isnone( FileName )):
        print( "Dataset file name has to be provided" )
        return None
    
    print( "Estimating optimal data mapping range to 1% saturation." )
    
    dtype = GetDataType( FileName = FileName )
          
    if( dtype == DataType.Tiling ):
        print( "Crating dummy mosaic..." )
        # TODO: This does not require ij indices. Remove requirement.
        if( isnone( D ) or isnone( mask ) or isnone( ij )):
            [ D, mask, ij ] = GetTilingData( D = D, FileName = FileName )
            
        I = MakeMosaicImage( D, mask, ij, remap = False )
        
    print( "Determining optimal mapping range..." )
    rng = get_frac_sat_rng( I )
        
    print( "Mapping range: [ %d , %d ]" %( rng[0], rng[1]))
    
    return rng

def GetTilingData( D = None, FileName = None ):
    """
    Get tiling data, mask and ij indices.
    """
    if( isnone( D )):
        print( "Reading data..." )
        D = ReadBinaryDataFile( FileName )
    
    config = ReadConfig( FileName )
    mask = GetIndexMask( config, 2 )
    
    ij = GetTileIJIndices( config = config )
    
    return [ D, mask, ij ]


def TileImages( D = None, FileName = None,
               img_sz = [ 780, 780 ], step_sz = [ 510, 525 ],
               rng = None, save_images = True,
               rng_override = None ):
    """
    Arrange images into a tiled mosaic while blindly accounting for tile
    overlap using tile_img_blind from lklib.image.
    
    Arguments:
        FileName - file name of data to tile
        img_sz - image size as [Y, X] in um
        step_sz - tiling grid spacing as [Y, X] in um
        rng - image value display mapping range as [min, max]
    """
    
    print( "Getting tiling data..." )
    [ D, mask, ij ] = GetTilingData( D = D, FileName = FileName )
    
    print( "Tiling..." )
    M = tile_img_blind( D = D, mask = mask, ij = ij,
                     img_sz = img_sz, step_sz = step_sz, rng = rng,
                     rng_override = rng_override )
    
    # crop the righ-hand side scan artefact which is not removed during blind
    # tiling
    crop_px = GetScanArtefactSize( FileName = FileName )
    M = M[ :, :-crop_px ]
    
    print( "Displaying..." )
    plt.clf()
    plt.imshow(M)
    
    if( save_images ):
        print( "Writing tiled images..." )
        save_img( M, ImageName = RemoveFileExtension(FileName) + "Tiled",
                  cmap = [ "viridis", "Greys" ], bake_cmap = True )

    print( "All done." )


def ExportChannels(config = None, D = None, ch_id = None, rng = None ):
    if(isnone(config)):
        raise Exception("NoConfig")
        
    if(isnone(ch_id)):
        ch_id = GetDefaultChannelIndex(config)
    
    idx_arr = GetIndexMask(config,ch_id)
    sat_thr = GetPixelCountLimit(config)
    bgr_lvl = GetPixelBackgroundCount(config)
    
    if(isnone(rng)):
        rng = [0,sat_thr]
          
    for ind in range(len(idx_arr)):
        data = D[:,:,idx_arr[ind]]
        num_sig = np.sum(data>bgr_lvl)
        num_sat = np.sum(data>sat_thr)
        num_over = np.sum(data>rng[1])
        img_file = r".\img\img_{:d}_{:d}.png".format(ch_id, ind)
        print("Saving file " + img_file + ". Signal: {:d} px, sat: {:d} px, over: {:d} px".format(num_sig, num_sat, num_over))
        plt.imsave(img_file, data , vmin=rng[0], vmax=rng[1], cmap="gray")
    

def ReadStabilityTrace( FileName ):
    
    if( isarray( FileName )):
        for ind in range( 0, len( FileName )):
            T, I = ReadStabilityTrace( FileName[ ind ])
            
            if( ind == 0 ):
                T_arr = [ T ]
                I_arr = [ I ]
            else:
                T_arr.append( T )
                I_arr.append( I )
                
        return [ T_arr, I_arr ]
            
    
    if( FileName.find( 'SigMon' ) != -1 ):
        D = ReadBigFile( FileName )
    elif( FileName.find( 'PowerLog' ) ):
        D = ReadStarLabFile( FileName )
    else:
        print( 'Could not determine file type' )
        return None
            
        
    #T = D[:,0] - D[0,0]
    T = D[:,0]
    I = D[:,1]
    
    # Handle overflows when time is reset to zero
    for i, t in enumerate(T):
        if t == 0 and i != 0:
            Tstep_pre_samples = T[i-10:i-1] - T[i-11:i-2]
            Tstep_post_samples = T[i:i+10] - T[i+1:i+11]
            
            if Tstep_pre_samples.std() > 1E-10:
                print( "Pre samples spaced unequally" )
            
            if Tstep_post_samples.std() > 1E-10:
                print( "Post samples spaced unequally" )
                
            T[i] = T[i-1] + Tstep_pre_samples.mean()*0.5 + Tstep_post_samples.mean()*0.5
            
            T[i+1:] = T[i+1:]+T[i]
            
        
    T = T - T[0]
    
    return [ T, I ]


def FormatTrace( T, I, T_ofs = None, T_scale = None,
                T_start = None, T_dur = None,
                baseline_corr = None, sub_mean_y = None ):
    
    if( not isnone( T_ofs )):
        T = T + T_ofs
        
    if( not isnone( T_scale )):
        T = T*T_scale
        
    if( not isnone( T_start )):
        ind = np.nonzero( T > T_start)[0][0]
        T = T[ind:]
        I = I[ind:]
        
    if( not isnone( T_dur )):
        markers = np.nonzero( T > T[0] + T_dur)
        if( len( markers[0] ) > 0 ):
            ind = markers[0][0]
            T = T[0:ind]
            I = I[0:ind]
        
    if( issmth( baseline_corr )):
        I = I/np.polyval( baseline_corr, T )
        
    if( sub_mean_y ):
        I = I - I.mean()
        
    return [ T, I ]


def PlotStabilityTrace( Tr = None,
                       T = None, I = None,
                       reduce_data = True,
                       FileName = None,
                       descr = None,
                       X_markers = None, plot_norm_trace = None,
                       show_zero_line = True,
                       plot_exp_sd = True,
                       xlim = None, ylim = None,
                       trace_ylim = None, hist_ylim = None, nf_ylim = None,
                       data_type = 'Counts', title = None, show_x_label = True,
                       ax_trace = None, ax_hist = None,
                       ax_noisefac = None, ax_stats = None,
                       show_trace = True, show_hist = True,
                       show_noisefac = False, show_stats = False,
                       show_stab_stats = None,
                       **kwargs ):
    """
    Plot a nice stability trace with data reduction and a histogram. Show the
    envelope of all values in gray, the average trace in blue and the expected
    std. dev. bounds in red.
    """
    
    try:
        if( FileName != None ):
            [ T, I ] = ReadStabilityTrace( FileName )
    
        if( not isnone( Tr )):
            [ T, I ] = Tr.GetTraceData( **kwargs )
            data_type = Tr.data_type
            title = Tr.title
            
            ylabel = Tr.Y_label
            
            if( isnone( plot_norm_trace )):
                plot_norm_trace = Tr.Y_norm
            
            if( plot_norm_trace ):
                ylabel = ylabel + ', a.u.'
            else:
                ylabel = ylabel + ', ' + Tr.Y_units               
            
        else:
            [ T, I ] = FormatTrace( T, I, **kwargs )
            
        if( isnone( xlim )):
            xlim = [ min(T), max(T) ]
            
        if( isnone( plot_norm_trace )):
            plot_norm_trace = False
    
        if( isnone( ax_trace ) and isnone( ax_hist ) and isnone( ax_noisefac ) and isnone( ax_stats ) ):
            if( show_noisefac == False and show_stats == False ):
                num_grid_rows = 1
            else:
                num_grid_rows = 2
                
            if( show_hist == True ):
                num_grid_cols = 3
            else:
                num_grid_cols = 2
                
    
            grid = plt.GridSpec(num_grid_rows, num_grid_cols, wspace=0.5, hspace=0.5)
            
            ax_trace = plt.subplot(grid[0,0:2])
            
            if( show_hist ):
                ax_hist = plt.subplot(grid[0,2])
            
            if( show_noisefac ):
                ax_noisefac = plt.subplot(grid[1,0:2])
                
            if( show_stats ):
                ax_stats = plt.subplot( grid[1, 2])
                
        Im = I.mean()
        
        # Determine expected standard deviation of the signal due to noise
        if( data_type == 'd' ):
            # For discrete data (e.g. photon counts) use shot-noise which is the
            # square root of the mean
            I_exp_sd = np.sqrt( Im )
        elif( data_type == 'c' ):
            # For continuous data (e.g volts) assume a fixed noise level
            I_exp_sd = 0.01
            
        if( plot_norm_trace ):
            I_p = I/Im
            Im_p = 1
            I_exp_sd_p = I_exp_sd/Im
        else:
            I_p = I
            Im_p = Im
            I_exp_sd_p = I_exp_sd
            
        if( isnone( ylim ) and not isnone( Tr ) ):
            if( not isnone( Tr.ylim )):
                ylim = Tr.ylim
            
        if( not isnone( ylim )):
            trace_ylim = ylim
            hist_ylim = ylim
        
        if( show_trace ):
            plt.axes( ax_trace )
            
            if( show_zero_line ):
                plt.plot( xlim, [ 0, 0 ], color = get_colour( "black" ) );
            
            [ T2, I2, I2_sd ] = plot_trace( T, I_p, marker = '.', reduce_data = reduce_data )
            if( plot_exp_sd ):
                plt.plot( [ T2[0], T2[-1] ], [ Im_p-I_exp_sd_p, Im_p-I_exp_sd_p ], color = get_colour( "darkred" ) )
                plt.plot( [ T2[0], T2[-1] ], [ Im_p+I_exp_sd_p, Im_p+I_exp_sd_p ], color = get_colour( "darkred" ) )
            plt.ylim( trace_ylim )
            
            if( not isnone( title )):
                plt.title( title )
            else:
                plt.title( "Stability trace" )
                
            if( show_x_label ):
                plt.xlabel( "Time (s)" )
                
            plt.ylabel( ylabel )
            
            if( not isnone( X_markers ) ):
                for xm in X_markers:
                    plt.plot( [ xm, xm ], plt.ylim(), color = get_colour( "darkgreen" ))
                    
            if( not isnone( Tr )):
                if( not isnone( Tr.ref_val )):
                    plt.plot( plt.xlim(), [ Tr.ref_val, Tr.ref_val ], color = get_colour( "gray" ))
                    
            plt.xlim( xlim )
            
            if( show_stab_stats ):
                # Print trace stability statistics in the upper left corner of the
                # plot axes
                
                s = ''
                
                # Add mean stability string, but only if the trace doesn't have
                # mean level substracton
                if( not isnone( Tr ) and not Tr.sub_mean_y ):
                    s = s + 'Mean = %.3f' %( I_p.mean() )
                
                # Format mean and std.dev. strings
                if( not( s == '' )):
                    s = s + ', '
                    
                s = s + 'sd = %.3f' %( I_p.std() )
                
                # Add fractional stability string, but only if the trace doesn't
                # have mean level subtraction
                if( not isnone( Tr ) and not Tr.sub_mean_y ):
                    s = s + ', stab = %.3f' %( I_p.std()/I_p.mean() )
    
                xlim = plt.xlim()
                ylim = plt.ylim()
                
                plt.text( xlim[0] + 0.02*(xlim[1]-xlim[0]),
                         ylim[1] - 0.02*(ylim[1]-ylim[0]),
                         s,
                         horizontalalignment='left', verticalalignment='top' )
        
        if( show_hist ):
            plt.axes( ax_hist )
            
            if( data_type == "Counts" ):
                bins = range( int(round(I.min())), int(round(I.max())) )
            elif( data_type == "Volts" ):
                bins = np.arange( I.min(), I.max(), 0.00016 )
                
            if( plot_norm_trace ):
                bins = bins/Im
                plt.hist( I/Im, bins = bins, orientation="horizontal" )
            else:
                plt.hist( I, bins = bins, orientation="horizontal" )    
                
            plt.ylim( hist_ylim )
            
            
        if( show_noisefac ):
            plt.axes( ax_noisefac )
            
            # Bin data into 1 second bins
            [ Tb, Ib, Ib_sd ] = reduce_trace( T, I, int((T[-1] - T[0])/1) )
            
            if( data_type == "Counts" ):
                nf = np.sqrt( Im )/Ib_sd
            elif( data_type == "Volts" ):
                nf = 0.01/Ib_sd
                    
            nf_m = nf.mean()
            nf_std = nf.std()    
            
            plot_trace( Tb, nf, color = get_colour( "darkblue" ))
            plt.plot( [ Tb[0], Tb[-1] ], [ nf_m, nf_m ], color = get_colour( "darkred" ) )
            plt.plot( [ Tb[0], Tb[-1] ], [ 1, 1 ], color = 'k' )
            plt.ylim( nf_ylim )
            plt.xlabel( "Time (s)" )
            plt.ylabel( "Noise factor" )
        
        
        if( show_stats ):
            plt.axes( ax_stats )
            plt.axis('off')
            
            if( descr == None and FileName != None ):
                descr = GetHeaderValue( ReadConfig( FileName ), "Scan Info", "Description" )
                
            srate = 1/(T[1] - T[0])
                
            s = ''
            
            if( descr != None ):
                s = s + descr + '\n'
                
            s = s + "Test duration: %.1f s\n" %( T[-1] - T[0] ) + \
                    "Sample rate: %.1f kHz\n" %( srate/1E3 )
                     
            if( data_type == "Counts" ):
                s = s + "Mean count rate: %.3f Mcps\n" %( Im*srate/1E6 )
            elif( data_type == "Volts" ):
                s = s + "Mean signal: %.3f V\n" %( Im )
                
            s = s + "Noise factor bin size: %.3f s\n" %( Tb[1] - Tb[0] ) + \
                    "Mean noise factor: %.3f ± %.3f\n" %( nf_m, nf_std ) + \
                    "Min/max noise factor: %.3f, %.3f" %( nf.min(), nf.max() )
                     
            plt.text( 0, 1, s, verticalalignment='top' )
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print( exc_type, fname, exc_tb.tb_lineno)
        print( "Could not plot trace" )
        
        
def CompareTraces( Trs = None,
                  T_arr = None, I_arr = None,
                  FileNames = None,
                  data_types = None, T_ofs_arr = None, T_scale_arr = None,
                  **kwargs ):
    
    Tr = None
    FileName = None
    T = None
    I = None
    T_ofs = None
    T_scale = None
    data_type = None
    show_hist = True
    
    
    for key, value in kwargs.items():
        if( key == 'show_hist' ):
            show_hist = value
    
    if( not isnone( Trs )):
        numR = len( Trs )
    elif( not isnone( FileNames )):
        numR = len( FileNames )
    else:
        numR = len( T_arr )
        
    if( show_hist ):
        grid = plt.GridSpec( numR, 3, wspace=0.5, hspace=0.5 )
    else:
        grid = plt.GridSpec( numR, 2, wspace=0.5, hspace=0.5 )
    
    for ind in range( 0, numR ):
        ax_trace = plt.subplot( grid[ind, 0:2] )
        if( show_hist ):
            ax_hist = plt.subplot( grid[ind, 2] )
        else:
            ax_hist = None
            
        if( not isnone( Trs )):
            Tr = Trs[ ind ]
        
        if( not isnone( FileNames )):
            FileName = FileNames[ ind ]
            
        if( not isnone( T_arr )):
            T = T_arr[ ind ]
            
        if( not isnone( T_ofs_arr )):
            T_ofs = T_ofs_arr[ ind ]
            
        if( not isnone( T_scale_arr )):
            T_scale = T_scale_arr[ind]
            
        if( not isnone( I_arr )):
            I = I_arr[ ind ]
            
        if( not isnone( data_types )):
            data_type = data_types[ ind ]
            
        if( ind == numR-1 ):
            show_x_label = True
        else:
            show_x_label = False
            
        PlotStabilityTrace( Tr = Tr, FileName = FileName, T = T, I = I, T_ofs = T_ofs, T_scale = T_scale,
                           ax_trace = ax_trace,
                           ax_hist = ax_hist, data_type = data_type,
                           show_x_label = show_x_label,
                           **kwargs )
        
        
def PlotCombinedStabilityTraces( T = None, I = None, Tc = None, Ic = None, FileNames = None, descr = None ):
    
    T_splice = []
    if( isnone( Tc ) and isnone( Ic ) ):
        numF = len( FileNames )
        
        for indF in range( 0, numF ):
            FileName = FileNames[ indF ]
            
            print( "Reading file " + str( indF ) )
            [ T, I ] = ReadStabilityTrace( FileName )
                    
            if( indF == 0 ):
                Tc = T
                Ic = I
            else:
                T_splice.append( Tc[-1] )
                Tc = np.append( Tc, T + Tc[-1] )
                Ic = np.append( Ic, I )
            
    PlotStabilityTrace( Tc, Ic, descr = descr, X_markers = T_splice )
    

def MakeCompositeImage( FileNames, method = "CombineToRGB", ofs = None, chas = None, corr_fi = True ):
    """
    Make a composite RGB image.
    """
    
    if( method == "CombineToRGB" ):
        numCh = len( FileNames )
        for ind in range( 0, numCh ):
            
            D = MakeImage( FileNames[ind], ch = 2, corr_fi = corr_fi )
            
            if( ind == 0 ):
                [ numR, numC ] = D[2].shape
                I = np.ndarray( [numR, numC, 3] ) # RGB output image
                I_raw = np.ndarray( [numR, numC, numCh])
                I_scaled = np.ndarray( [numR, numC, numCh] )
                cm = []
                rng = []
                gamma = []
                
            if( issmth(chas) ):
                ch_ind = chas[ind]
            else:
                ch_ind = ind                    
                
            I_raw[:, :, ind] = D[1]
            I_scaled[:, :, ind] = D[2]/255
            cm.append( D[3])
            rng.append( D[4])
            gamma.append( D[5])
            
            ofs_XY = ofs[ind]
            
            if( ofs_XY != None ):
                ofs_X = ofs[ind][0]
                ofs_Y = ofs[ind][1]
                I[:-ofs_Y, :-ofs_X, ch_ind] = I_scaled[ofs_Y:, ofs_X:, ind]
            else:
                I[:, :, ch_ind] = I_scaled[:, :, ind]
                    
        return [ I, I_raw, I_scaled, cm, rng, gamma ]
            
    elif( method == "BlendRGB"):
        print( "RGB blending is not yet working" )
        #I0 = (D0[0])[:,:,0:3]
        #a0 = D0[1].astype(float)
        #I1 = (D1[0])[:,:,0:3]
        #a1 = D1[1].astype(float)
        
        #a0 = a0/a0.max()
        #a1 = a1/a1.max()
        
        #a0 = a0**0.5
        
        #a0 = a0/(a0+a1)
        #a1 = a1/(a0+a1)
        
        #I = .alpha_blend( I0, I1, a1=a0, a2=a1 )
        
        #scipy.misc.imsave('I0.png', I0)
        #scipy.misc.imsave('I1.png', I1)
        #scipy.misc.imsave('I.png', I)
        
        return I
    else:
        print( "Unknown method" + method ) 
        
    return None


def GenerateImageReport( I = None, D = None, FileName = None, rng = None,
                        chan_ind = None, gamma = None,
                        chas = None,
                        plot_raw_hist = True, plot_mapped_hist = True,
                        plot_sat_map = True, export_figure = True, fig_suffix = '', 
                        corr_fi = True, cm = None, cm_sat = False,
                        write_image = True, write_unprocessed_grayscale = False ):
    
    config = ReadConfig( FileName )
    
    if(isnone(chan_ind)):
        chan_ind = GetDefaultChannelIndex(config)
        print("Channel index not specified assuming ch_ind=%d" %(chan_ind))
        
    ValidateChannelIndex(config, chan_ind)
    chan_type = GetChannelDetectorType(config, chan_ind)
   
    if( isnone( config )):
        print( "Could not obtain config data, cannot generate image." )
        return
    
    if( type( FileName ) == type(str()) ):
        composite = False
        title_str = MakeImageTitle( config, chan = chan_ind, print_exw = True )
        [ I, I_raw, I_scaled, cm, rng, gamma ] = MakeImage( I = I, D = D, FileName = FileName, rng = rng, gamma = gamma, ch = chan_ind, corr_fi = corr_fi, cm = cm, cm_sat = cm_sat )
    else:
        composite = True
        title_str = MakeImageTitle( config, chan = chan_ind, print_exw = True, chas = chas )
        [ I, I_raw, I_scaled, cm, rng, gamma ] = MakeCompositeImage( FileName, ofs = [None,None,None], chas = chas)
    
    
    [ I, scalebar_sz ] = add_scale_bar( I, pxsz=GetScanPixelSize( config ))
    
    grid = plt.GridSpec( 2, 4 )
    if( plot_raw_hist == False and plot_mapped_hist == False and plot_sat_map == False ):
        plt.subplot(grid[0:2,0:4])
    else:
        plt.subplot(grid[0:2,0:2])
        
    
    show_img( I, title = title_str, remap = False )
    
    if( write_image ):
        mpimg.imsave( FileName[ :FileName.rfind('.') ] + 'img' + '.png', I )
        
    if(write_unprocessed_grayscale):
        plt.imsave( FileName[ :FileName.rfind('.') ] + 'img_u' + '.png',
                             I_raw, vmin=rng[0], vmax=rng[1], cmap="gray")
    
    if(chan_type==DetectorType.Counter):
        img_stats=CountImageStats()
        if( composite ):
            img_stats.TotalCount = np.empty_like( gamma )
            img_stats.MaxCount = np.empty_like( gamma )
            for indCh in range( 0, len( config )):
                img_stats.TotalCount[indCh] = I_raw.sum()
                img_stats.MaxCount[indCh] = I.max()
        else:
            img_stats.TotalCount = I_raw.sum()
            img_stats.MaxCount = I_raw.max()
    elif(chan_type==DetectorType.Voltage):
        img_stats=VoltageImageStats()
        img_stats.MinLevel = np.min(I_raw)
        img_stats.AvgLevel = np.mean(I_raw)
        img_stats.MaxLevel = np.max(I_raw)
        
    caption_str = MakeCaptionString( config, ch_ind = chan_ind,
                                    rng = rng, gamma = gamma, cm = cm,
                                    scalebar_sz = scalebar_sz,
                                    image_stats = img_stats )
    
    numR = I.shape[0]
    plt.text( 0, numR*1.02, caption_str, verticalalignment='top' )
    
    if( plot_raw_hist ):
        plt.subplot(grid[0,2])
        plt.hist( I_raw.flatten(), bins = 256, log = True )
        ax = plt.gca()
        ax.set_title( "Raw histogram" )
    
    if( plot_mapped_hist ):
        plt.subplot(grid[0,3])
        plt.hist( I_scaled.flatten(), bins = 256, log = True )
        ax = plt.gca()
        ax.set_title( "Mapped histogram" )
    
    if( plot_sat_map and composite == False ):
        satM = GetSatMask( I_raw, config )
        if( isnone( satM ) == False ):
            plt.subplot(grid[1,2])
            show_img( satM/4, cmap = get_colourmap( "GYOR_Nice" ), remap = False )
            ax = plt.gca()
            ax.set_title( "Saturation map" )
            if( (satM>1).any() == False ):
                plt.text( 0, satM.shape[0]*1.05, "No saturation" )
            else:
                sat1 = (satM>1).sum()/len(satM.flatten())
                if( sat1 > 0.001 ):
                    sat1_str = "%.3f" %( (satM>1).sum()/len(satM.flatten()))
                else:
                    sat1_str = str( (satM>1).sum() ) + " px"
                    
                plt.text( 0, satM.shape[0]*1.05, 
                         "Saturation: >1 "+ sat1_str + "; "
                         ">2 "+ "%.3f" %( (satM>2).sum()/len(satM.flatten())) + "; "
                         #">3 "+ "%.3f" %( (satM>3).sum()/len(satM.flatten())) + "; "
                         ">4 "+ "%.3f" %( (satM>4).sum()/len(satM.flatten())))
    
    #else:
    #    if D_type == lk.DataType.TimeLapse:
    #        mos_type = lk.MosaicType.TimeSeries
    #    elif D_type == lk.DataType.ZStack:
    #        mos_type = lk.MosaicType.ZStack
    #    else:
    #        print( "Unknown data type" + str( D_type ))
    #        #return None
    #
    #    lk.ShowMosaic( D, FileName, mos_type=mos_type )
    
    if( export_figure == True ):
        if( composite ):
            export_figure( FileName[0], suffix = fig_suffix + "comb" )
        else:        
            export_figure( FileName, suffix = fig_suffix )
            
            
def GenerateOutputImages( FileName = None, D = None, step_sz = None, rng = None, rng_override = None,
                         make_basic_report_fig = True, make_detailed_report_fig = True, write_grayscale_img = False ):
    try:
        config = ReadConfig( FileName )
        
        dtype = GetDataType( config = config )
        
        if( isnone( dtype )):
            print( "Could not determine data type" )
            raise Exception( "InvalidDataType" )
    
        if( dtype == DataType.Tiling ):
            # TODO: scan field size calibration is out of date. Fix it.
            img_sz = GetScanFieldSize( config, apply_sz_calib = False )
            img_sz = [ img_sz, img_sz ]
            
            if( isnone( step_sz )):
                step_sz = GetTilingStep( config )*1000
                step_sz = [ step_sz, step_sz ]
                
            [ D, mask, ij ] = GetTilingData( FileName = FileName, D = D )
            
            if( isnone( rng )):
                rng = GetOptimalMappingRange( D = D, FileName = FileName, mask = mask, ij = ij )
            
            print( "Making raw tiled image..." )
            ShowRawTiledImage( FileName = FileName, D = D, rng = rng )

            TileImages( D = D, FileName = FileName, img_sz = img_sz, step_sz = step_sz, rng = rng, rng_override = rng_override )
        else:
            [ I, rng, gamma, D ] = ProcessImage( FileName = FileName )
            if( make_detailed_report_fig ):
                plt.figure(1)
                GenerateImageReport( I = I, D = D, FileName = FileName, fig_suffix = "detailed_fig", corr_fi = False, rng = rng, gamma = gamma )
                
            if( make_basic_report_fig ):
                plt.figure(2)
                GenerateImageReport( I = I, D = D, FileName = FileName, fig_suffix = "basic_fig", plot_raw_hist = False, rng = rng, gamma = gamma,
                            plot_mapped_hist = False, corr_fi = False, plot_sat_map = False )
                
            if( write_grayscale_img ):
                I_save = np.round( ( I - rng[0] )/( rng[1]-rng[0])*255 )
                I_save[ I_save>255 ] = 255
                save_img( I_save.astype( np.uint8 ), ImageName = RemoveFileExtension( FileName ), suffix = "bw", img_type = "png", cmap = "gray" )
            
    except:
        handle_general_exception( "Could not generate output images for file " + FileName )
        

def GetStabilityMeasurementStartTimeAndDur( Tofs_ts, DataDir = None ):
    """
    Get the start time and duration of the stability measurement stored in a
    TDMS file. Measurement start and stop times are expected to be stored as
    timestamps in Timing.ini.
    """
    
    # Read sync times
    if( isnone( DataDir )):
        DataDir = r".\\"
        
    timing_file = DataDir + r"\Timing.ini"
    Cfg = ReadConfig( timing_file )
    d = datetime.strptime( GetHeaderValue( Cfg, 'Timing', 'Start Sync' ), '%Y.%m.%d %H:%M:%S.%f' )
    T_start_ts = time.mktime( d.timetuple() )

    d = datetime.strptime( GetHeaderValue( Cfg, 'Timing', 'End Sync' ), '%Y.%m.%d %H:%M:%S.%f' )
    T_end_ts = time.mktime( d.timetuple() )

    # Read THG microscope signal, laser avg. power and avg. peak intensity traces
    [ T, I, Tofs_ts ] = ReadTDMSTraces( DataDir )

    Tofs_ts = time.mktime( Tofs_ts.timetuple() )

    # Timing.ini and TDMS timestamps have different timezone handling. Subtract
    # two hours from both values to align.
    # Due to the delay in opening and closing the shutter even when using
    # blocking calls the trace is skewed in time with respect to sync
    # timestamps. Cut 1 s from the beginning of the trace and 3 s from the end
    # to make sure the shutter sync wiggles stay outside of the the stability
    # measurement range.
    T_start = T_start_ts - Tofs_ts - 7200 + 1
    T_dur = T_end_ts - Tofs_ts - 7200 - T_start - 3
    
    return [ T_start, T_dur ]


def GetStabTraces( DirName, CropT = True, T_start = None, T_dur = None, Scaled3 = False, **kwargs ):
    # Read THG microscope signal, laser avg. power and avg. peak intensity traces
    [ T, I, TDMS_ofs_ts ] = ReadTDMSTraces( DirName )

    [ T_start_ts, T_dur_ts ] = GetStabilityMeasurementStartTimeAndDur( TDMS_ofs_ts, DataDir = DirName )
    
    if( isnone( T_start )):
        T_start = T_start_ts
        
    if( isnone( T_dur )):
        T_dur = T_dur_ts
        
    THG = Trace( T = T[0], Y = I[0], Y_label = 'Counts', title = 'THG', data_type = 'd' )
    
    if( Scaled3 ):
        P_Avg = Trace( T = T[0], Y = I[1]**3, Y_label = 'Volts', title = 'Avg. Power Ref. ^3', data_type = 'c' )
        I_Avg = Trace( T = T[0], Y = I[2]**3/2, Y_label = 'Volts', title = 'SHG Ref. ^3/2', data_type = 'c' )
    else:
        P_Avg = Trace( T = T[0], Y = I[1], Y_label = 'Volts', title = 'Avg. Power Ref.', data_type = 'c' )
        I_Avg = Trace( T = T[0], Y = I[2], Y_label = 'Volts', title = 'SHG Ref.', data_type = 'c' )

    if( CropT ):
        trace_set_param( [ THG, P_Avg, I_Avg ],
                    T_ofs = 0,
                    T_start = T_start, T_dur = T_dur )
    
    return [ THG, P_Avg, I_Avg ]
        
        
def GenStabReport( DataDir, Type = None, ylim_norm = None,
                   show_png_ext = False, copy_fig_to_storage = False,
                   **kwargs ):
    
    try:
        # Add a trailing backslash to the directory path if it is not given
        if( DataDir[-1] != '\\' ):
            DataDir = DataDir + '\\'
            
        args = { 'T_start': kwargs.get( 'T_start' ), 'T_dur': kwargs.get( 'T_dur' ) }

        # Read THG, avg. power and peak intensity traces         
        if( Type == "THG_Avg_Peak" ):
            [ THG, P_Avg, I_Avg ] = GetStabTraces( DataDir, Scaled3 = True,
                **args )
            
            Trs = [ THG, P_Avg, I_Avg ]
            trace_set_param( Trs, Y_norm = True, ylim = ylim_norm )
            fig_file_name = "THG_vs_avg_and_peak.png"
            
        elif( Type == "OscPwr" or Type == "OscBarTemp" or Type == "OscBarV" or Type == "OscBarI" or Type == "Ophir" or Type == "THG_vs_nearXY" or Type == "THG_vs_farXY" ):
            [ THG, P_Avg, I_Avg ] = GetStabTraces( DataDir, **args )
            P_Avg.title = "Average Power (diode)"
            P_Avg.Y_norm = True
            P_Avg.ylim = ylim_norm
            
        # Read Pharos power log traces
        if( Type == "OscPwr" ):
            T_ofs_ph = kwargs.get( "T_ofs_ph" )
            P_avg_osc = get_pharos_log_trace( DirName = DataDir, DataType = Type, T_ofs = T_ofs_ph, **args )
            P_avg_osc.Y_norm = True
            P_avg_osc.ylim = ylim_norm
            Trs = [ P_avg_osc, P_Avg ]
            fig_file_name = "FLINT_vs_diode.png"
        elif( Type == "OscBarTemp" ):
            T_ofs_ph = kwargs.get( "T_ofs_ph" )
            Temp_bar = get_pharos_log_trace( DirName = DataDir, DataType = Type, T_ofs = T_ofs_ph, **args )
            Trs = [ Temp_bar, P_Avg ]
            fig_file_name = "OscBarTemp_vs_diode.png"
        elif( Type == "OscBarV" ):
            T_ofs_ph = kwargs.get( "T_ofs_ph" )
            Temp_bar = get_pharos_log_trace( DirName = DataDir, DataType = Type, T_ofs = T_ofs_ph, **args )
            Trs = [ Temp_bar, P_Avg ]
            fig_file_name = "OscBarV_vs_diode.png"
        elif( Type == "OscBarI" ):
            T_ofs_ph = kwargs.get( "T_ofs_ph" )
            Temp_bar = get_pharos_log_trace( DirName = DataDir, DataType = Type, T_ofs = T_ofs_ph, **args )
            Trs = [ Temp_bar, P_Avg ]
            fig_file_name = "OscBarI_vs_diode.png"
        elif( Type == "Ophir" ):
            T_ofs_ophir = kwargs.get( "T_ofs_ophir" )
            P_avg_ophir = read_tdms_trace_ophir( DirName = DataDir, T_ofs = T_ofs_ophir, **args )
            P_avg_ophir.title = "Average Power (Ophir)"
            P_avg_ophir.Y_norm = True
            P_avg_ophir.ylim = ylim_norm
            Trs = [ P_avg_ophir, P_Avg ]
            fig_file_name = "Ophir_vs_diode.png"
            
            
        # Read beam position traces
        if( Type == "THG_vs_nearXY" or Type == "THG_vs_farXY" or Type == "beam_pos" or Type == "beam_pos_ofs" ):
            T_ofs_lcbd = kwargs.get( "T_ofs_lcbd" )
            
            D = ReadLCBeamDiagnostics( GetLCBeamDiagnosticsPath( DataDir, 2 ) )
            NX = Trace( T = D[:,0], Y = D[:,1], title = 'Near Deviation X' )
            NY = Trace( T = D[:,0], Y = D[:,2], title = 'Near Deviation Y' )
            
            D = ReadLCBeamDiagnostics( GetLCBeamDiagnosticsPath( DataDir, 1 ) )
            FX = Trace( T = D[:,0], Y = D[:,1], title = 'Far Deviation X' )
            FY = Trace( T = D[:,0], Y = D[:,2], title = 'Far Deviation Y' )
        
        # Make beam position stability plots
        if( Type == "THG_vs_nearXY" ):
            NX.title = "Near Deviation X"
            NY.title = "Near Deviation Y"
            trace_set_param( [ NX, NY ],
                T_ofs = T_ofs_lcbd, T_scale = 1,
                sub_mean_y = True, Y_label = 'Deviation', Y_units = 'um',
                data_type = 'c',
                ref_val = 0, ylim = [ -30, 30 ],
                **args )
        elif( Type == "THG_vs_farXY" ):
            FX.title = "Far Deviation X"
            FY.title = "Far Deviation Y"
            trace_set_param( [ FX, FY ],
                T_ofs = T_ofs_lcbd, T_scale = 1,
                sub_mean_y = True, Y_label = 'Deviation', Y_units = 'um',
                data_type = 'c',
                ref_val = 0, ylim = [ -30, 30 ],
                **args )
            
            Trs = [ THG, FX, FY ]
            fig_file_name = "THG_vs_farXY.png"
            
        elif( Type == "beam_pos" ):
            NX.title = "Near Position X"
            NY.title = "Near Position Y"
            
            FX.title = "Far Position X"
            FY.title = "Far Position Y"

            trace_set_param( [ NX, NY, FX, FY ],
                T_ofs = T_ofs_lcbd, T_scale = 1,
                Y_label = 'Position', Y_units = 'um',
                data_type = 'c',
                **args )
            
            Trs = [ NX, NY, FX, FY ]
            fig_file_name = "beam_pos.png"
            
        elif( Type == "beam_pos_ofs" ):
            NX.title = "Near Offset X"
            NY.title = "Near Offset Y"
            NX.ref_val = 2784
            NY.ref_val = 1610
            
            FX.title = "Far Offset X"
            FY.title = "Far Offset Y"
            FX.ref_val = 2410
            FY.ref_val = 1529

            trace_set_param( [ NX, NY, FX, FY ],
                T_ofs = T_ofs_lcbd, T_scale = 1,
                sub_ref_val = True,
                Y_label = 'Offset', Y_units = 'um',
                data_type = 'c',
                **args )
            
            Trs = [ NX, NY, FX, FY ]
            fig_file_name = "beam_pos_ofs.png"
            
            
            
        CompareTraces( Trs = Trs,
                      show_stab_stats = True,
                      plot_exp_sd = False,
                      show_hist = False,
                      **kwargs )
        
        fig_file_path = DataDir + fig_file_name
        export_figure( fig_file_path )
        
        if( show_png_ext ):
            show_png_ext( fig_file_path )
            
        if( copy_fig_to_storage ):
            CopyStabFigToStorage( DataDir, fig_file_path )
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print( exc_type, fname, exc_tb.tb_lineno)
        print( "Could not analyze trace" )
        input()
        
def CopyStabFigToStorage( DataDir, FigFileName ):
    s = DataDir
    date_str = re.findall(r"(\d{4}-\d{2}-\d{2})", s)[0]
    
    dst = r"Z:\Projects\LCM\Data\Signal Stability\Stability Traces\\" + date_str + ".png"
    
    copyfile( FigFileName, dst )
    
def GenerateReport( file_name = None, img_file_names = None, chan_id = 2, dry_run = False ):
    
    if( isnone( file_name )):
        raise ValueError( "File name not given" )
        
    if( isnone( img_file_names )):
        img_file_names = ListFilesWithFilter( RemoveFileExtension( file_name ) + 'Tiled_*' + '*.png' )
    
    for img_file_name in img_file_names:
        img_sz = get_img_sz( file_name = img_file_name )
    
        if( img_file_name.find( "viridis" ) != -1 ):
            img_cmap = "viridis"
        elif( img_file_name.find( "WK" ) != -1 or img_file_name.find( "Greys" ) != -1 ):
            img_cmap = "WK"
        else:
            img_cmap = img_file_name[ img_file_name.rfind( '_' )+1:-4 ]
            print( "Unknown colourmap for file ''%s'', using ''%s''." %( img_file_name, img_cmap ))
    
        config = ReadConfig( file_name )
                
        sample_id_str = GetSampleID( config )
        sample_area_label = GetSampleAreaLabel( config )
        
        sample_map_file_name = sample_id_str + '.svg'
        
        sample_map_href = sample_map_file_name + '#HE'
        sample_map_area_href = sample_map_file_name + '#' + sample_area_label
        
        rng = GetConfigRange( config, chan_id = chan_id )
        
        if( isnone( rng )):
            rng = GetOptimalMappingRange( FileName = file_name )

        um_px = GetScanPixelSize( config, apply_sz_calib = False )
        
        title_str = MakeImageTitle( config, template = "report", chan = chan_id )
            
        cap_str = MakeCaptionString( config, template = "report", rng =  rng,
                                    gamma = 1, cm = img_cmap, img_sz = img_sz )
            
        MakeSVGReport( img_file_name = img_file_name, img_sz = img_sz,
                        um_px = um_px, img_cmap = img_cmap,
                        title_str = title_str, cap_str = cap_str,
                        sample_map_href = sample_map_href,
                        sample_map_area_href = sample_map_area_href,
                        dry_run = dry_run )
        
        ConvertSVGToPDF( ChangeExtension( img_file_name, "svg" ), dry_run = dry_run )
    
def EstimateSpotSize(lam=1.028,NA=0.75,n=1):
    """
    Estimate the lateral and axial spot size.
    """
    w0_xy = 0.318*lam/NA
    Wh_xy = 1.18*w0_xy
    
    Wh_z = 0.61*lam/(n-np.sqrt(n**2-NA**2))
    
    print("NA = %.2f, lamda = %.3f um, n = %.3f" %(NA, lam, n))
    print("XY = %.2f um FWHM, Z = %.2f um FWHM" %(Wh_xy, Wh_z))


def CountLinearityCorrection(C, t=None, f=None):
    if(isnone(t)):
        print("Assuming 1 s integration time")
        t = 1
    if(isnone(f)):
        print("Assuming 75 MHz repetition rate")
        f = 75e6
        
    R = C/t
    print("Count rate: %.3f Mcps" %(R/1e6))
    
    P = R/f
    print("Count probability: %.3f" %P)
    
    f = 0.5*P**2
    print("Correction factor: %3g" %f)
    
    fsev = f/P
    print("Correction severity: %3g" %fsev)
    
    Cc = C*(1+fsev)
    print("Corrected counts: %.3f Mcps" %(Cc/1e6))
    
    print("Count delta: %.3f Mcps" %((Cc - C)/1e6))
    print("Correction severity: %.3f" %((Cc-C)/C))
