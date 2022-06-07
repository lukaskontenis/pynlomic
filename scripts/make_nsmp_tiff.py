"""Make a multipage NSMP TIFF image.

Convert an NSMP dataset to a multipage TIFF file for NLPS.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lkcom.util import handle_general_exception

from lcmicro.proc import convert_nsmp_to_tiff
from lcmicro.dataio import get_microscopy_data_file_name

print("=== NSMP TIFF converter ===")

file_name = None
num_args = len(sys.argv)
if num_args < 2:
    file_name = get_microscopy_data_file_name()
else:
    file_name = sys.argv[1]

if file_name is None:
    print("No input provided. Specify a file name using:")
    print("\t" + os.path.basename(__file__) + " scan.dat")
    print("\nOr drag a dat file on the script icon.\n")
else:
    try:
        print("Converting '{:s}' to TIFF...".format(file_name))

        convert_nsmp_to_tiff(file_name=file_name)

    except Exception:
        handle_general_exception("Conversion failed")

input("Pess any key to close this window...")
