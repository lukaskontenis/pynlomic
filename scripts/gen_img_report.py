"""Generate image report.

Convert a raw microscope data file to PNG and create an image report.

Args:
    file_name – raw data file name
    rng – min/max count range to map to [0, 255] output image levels
    gamma – gamma parameter of the mapping linearity

A header file with the same name as the data file and an .ini extension should
be present in the same directory as the data file.

The gamma parameter is useful when generating 8-bit images from high dynamic
range count data, especially for harmonic-generation microscopy images.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

file_name = '18_01_04_.629_.dat'
rng = [0, 22000]
gamma = 1

print("=== lcmicro ===")
print("Running image report generation script...")

from lklib.util import handle_general_exception
from lcmicro.report import gen_img_report

try:
    gen_img_report(file_name=file_name, corr_fi=False, rng=rng, gamma=gamma, chan_ind=3)
except Exception:
    handle_general_exception("Could not generate image report")

input("Pess any key to close this window...")
