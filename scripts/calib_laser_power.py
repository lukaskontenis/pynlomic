"""Calibrate laser power attenuator.

Calculates calibrtion parameters for a rotating-waveplate laser power
attenuator.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

file_name = r"D:\Data\Maintenance\LCM1\Power calib\2020-09-17\calibration_2020_09_17.txt"

print("=== lcmicro ===")
print("Running laser power calibration script...")

from lklib.util import handle_general_exception
from lcmicro.report import calib_laser_power

try:
    calib_laser_power(file_name)
except Exception:
    handle_general_exception("Could not perform calibration")

input("Press any key to close this window...")
