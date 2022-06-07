"""Make a PSF figure.

Make an axial THG Z PSF figure from a line scan data file.

If the trace does not reach zero due to background you can disable zero line
plotting by setting show_y_zero_marker to False.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os
import pathlib

from lkcom.util import handle_general_exception

from lcmicro.report import gen_thg_psf_fig
from lcmicro.dataio import get_psf_data_file_name

print("=== PSF figure generator ===")

file_name = None
num_args = len(sys.argv)
if num_args < 2:
    file_name = get_psf_data_file_name()
else:
    file_name = sys.argv[1]

if file_name is None:
    print("No input provided. Specify a file name using:")
    print("\t" + os.path.basename(__file__) + " psf.txt")
    print("\nOr drag a TXT file on the script.\n")
else:
    try:
        print("Generating PSF figure for '{:s}'...".format(
            pathlib.Path(file_name).absolute().name))

        gen_thg_psf_fig(
            file_name=file_name, wavl=1.03, show_y_zero_marker=False,
            suptitle_suffix='Second surface')
    except Exception:
        handle_general_exception("Conversion failed")

input("Pess any key to close this window...")
