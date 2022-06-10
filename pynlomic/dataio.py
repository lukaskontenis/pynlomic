"""pynlomic - a Python library for nonlinear microscopy.

Microscopy data input/output functions.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import os
import zipfile
from pathlib import Path

from lkcom.dataio import list_files_with_extension


def get_microscopy_data_file_name(file_name=None):
    """Get microscopy data file name in the current dir.

    This function returns the name of a microscopy data file, if there is only
    one in the current directory. It does that by listing .dat files in the
    current directory while skipping the PolStates.dat file.
    """
    file_names = list_files_with_extension(ext='dat')

    # Remove PolStates.dat files
    file_names2 = []
    for file_name in file_names:
        if os.path.basename(file_name).find('PolStates') == -1:
            file_names2.append(file_name)
    file_names = file_names2

    if len(file_names) == 0:
        zip_files = list_files_with_extension(ext='zip')

        for zip_file in zip_files:
            zipf = zipfile.ZipFile(zip_file)
            zip_contents = zipf.namelist()
            if len([name for name in zip_contents if Path(name).suffix=='.dat']) > 0 \
                    and len([name for name in zip_contents if Path(name).suffix=='.ini']) > 0:
                file_names.append(zip_file)

    if len(file_names) == 0:
        print("No data files found")
        return None
    if len(file_names) == 1:
        file_name = file_names[0]
        print("Found a single dat file '{:s}', loading it".format(file_name))
        return file_name
    else:
        print("More than one dat file found, specify which to load")
        return None


def get_psf_data_file_name(file_name=None):
    """Get PSF data file name in the current dir.

    This function returns the name of a PSF data file, if there is only
    one in the current directory. It does that by listing .txt files in the
    current directory.
    """
    file_names = list_files_with_extension(ext='txt')

    if len(file_names) == 0:
        print("No data files found")
        return None
    if len(file_names) == 1:
        file_name = file_names[0]
        print("Found a single TXT file '{:s}s', loading it".format(file_name))
        return file_name
    else:
        print("More than one TXT file found, specify which to load")
        return None
