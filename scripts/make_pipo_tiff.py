"""Make a multipage PIPO TIFF image.

Convert a PIPO dataset to a multipage TIFF file.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lklib.fileread import list_files_with_extension
from lklib.util import handle_general_exception

from lcmicro.proc import convert_pipo_to_tiff
from lcmicro.dataio import get_microscopy_data_file_name

print("=== PIPO to TIFF converter ===")

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

        convert_pipo_to_tiff(file_name)

    except Exception:
        handle_general_exception("Conversion failed")

input("Pess any key to close this window...")
