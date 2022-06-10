"""Calibrate laser power attenuator.

Calculates calibrtion parameters for a rotating-waveplate laser power
attenuator.

This script is part of pynlomic, a Python library for nonlinear microscopy.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

file_name = r"2022-04-20 - FF calib.txt"

print("=== pynlomic ===")
print("Running laser power calibration script...")

from lkcom.util import handle_general_exception
from pynlomic.report import calib_laser_power

try:
    calib_laser_power(file_name)
except Exception:
    handle_general_exception("Could not perform calibration")

input("Press any key to close this window...")
