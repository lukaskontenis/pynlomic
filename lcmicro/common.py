
"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains common enums and routines.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
from enum import Enum


class ImageStats:
    """Image statistics base class."""

    Type = None


class VoltageImageStats(ImageStats):
    """Voltage image statistics enum."""

    MinLevel = None
    AvgLevel = None
    MaxLevel = None


class CountImageStats(ImageStats):
    """Count image statistics enum."""

    TotalCount = None
    MaxCount = None


class DetectorType(Enum):
    """Detector type enum."""

    Counter = 1
    Voltage = 2


class MosaicType(Enum):
    """Mosaic time enum."""

    TimeSeries = 1
    ZStack = 2


class DataType(Enum):
    """Data file type enum."""

    Invalid = 0
    SingleImage = 1
    Average = 2
    TimeLapse = 3
    ZStack = 4
    Tiling = 5


class PixelCountLimit():
    """Pixel count limit struct."""

    RepRate = None
    SinglePulse = None
    CountLinearity = None


def get_data_type_str(dtype):
    """Return the name of the data type as a string."""
    if dtype == DataType.SingleImage:
        return "Single image"
    if dtype == DataType.Average:
        return "Average"
    if dtype == DataType.TimeLapse:
        return "Time lapse"
    if dtype == DataType.Tiling:
        return "Tiling"
    if dtype == DataType.ZStack:
        return "Z Stack"
    return "INVALID DATA TYPE"
