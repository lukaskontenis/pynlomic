"""Microscopy data input/output functions

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import os
from lklib.fileread import list_files_with_extension


def get_microscopy_data_file_name(file_name=None):
    file_names = list_files_with_extension(ext='dat')

    # Remove PolStates.dat files
    file_names2 = []
    for file_name in file_names:
        if os.path.basename(file_name) != 'PolStates.dat':
            file_names2.append(file_name)
    file_names = file_names2

    if len(file_names) == 0:
        print("No data files found")
        return None
    if len(file_names) == 1:
        file_name = file_names[0]
        print("Found a single dat file '{:s}s', loading it".format(file_name))
        return file_name
    else:
        print("More than one dat file found, specify which to load")
        return None
