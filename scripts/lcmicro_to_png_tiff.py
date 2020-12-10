"""Convert microscope data to PNG/TIFF.

Convert a raw microscope data file to PNG/TIFF.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os
from lcmicro.report import export_img_png_tiff
from lklib.util import handle_general_exception

print("=== lcmicro ===")
print("Running PNG/TIFF converter...")

num_args = len(sys.argv)
if num_args < 2:
    print("No input provided. Specify a file name using:")
    print("\t" + os.path.basename(__file__) + " scan.dat")
    print("\nOr drag a dat file on the script icon.\n")
else:
    try:
        file_name = str(sys.argv[1])
        print("Parsing file: " + file_name)
        export_img_png_tiff(file_name=file_name)
    except Exception:
        handle_general_exception("Exporting failed")

input("Pess any key to close this window...")
