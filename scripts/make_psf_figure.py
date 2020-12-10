"""Make PSF figure.

Make a THG Z PSF figure from a line scan data file.

If the trace does not reach zero due to background you can disable zero line
plotting by setting show_y_zero to False.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

print("=== lcmicro ===")
print("Generating PSF figure...")

from lklib.util import handle_general_exception
from lcmicro.report import gen_thg_psf_fig

try:
    gen_thg_psf_fig(
        file_name='800_PSF.txt', wavl=0.8, show_y_zero_marker=False,
        suptitle_suffix='Second surface, LCM1, CRONUS VIS beam')
except Exception:
    handle_general_exception("Could not generate figure")

input("Pess any key to close this window...")

