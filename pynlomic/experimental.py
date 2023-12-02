"""pynlomic - a Python library for nonlinear microscopy.

This module contains experimental features.

Copyright 2015-2023 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from lkcom.dataio import list_dirs, list_files_with_extension
from pynlomic.proc import load_pipo

def parse_custom_pipo_data():
    """Parse custom PIPO data.

    Make a standard PIPO dataset from custom PIPO data that cannot be handled
    by the main parser. This script was first created to handle PIPO datasets
    that were acquired without PSA automation, i.e., the microscope program had
    no means to control PSA angles and PIPO datasets were acquired with
    automated PSG state control and manually going over the PSA states. The
    result is a series of folders where each folder contains multiple PSG
    measurements at a single PSA setting.

    PSA folders are expected to be in current dir and named after the PSA
    setting. Each PSA folder contains a .dat file that can be handled by
    pynlomic.
    """
    print("=== Custom PIPO data parser ===")

    print("This script will parse a custom PIPO dataset from a series of "
          "pynlomic-copatible DAT files residing in PSA dirs.")

    print("Reading directories...")

    psa_dirs = list_dirs('.')
    if len(psa_dirs) < 1:
        raise RuntimeError("No directories found")

    # Figure out the directory format
    try:
        # Directories are just degrees
        psa_angles = [float(psa_dir) for psa_dir in psa_dirs]
    except Exception as excpt:
        psa_angles = None

    if psa_angles is None:
        try:
            # Directory namies have an explicit deg string, i.e., '<anlge> deg'
            psa_dirs = [psa_dir for psa_dir \
                        in psa_dirs if psa_dir.find('deg') != -1]
            psa_angles = [float(psa_dir.split('deg')[0]) \
                          for psa_dir in psa_dirs]
        except Exception as excpt:
            psa_angles = None

    if psa_angles is None:
        raise RuntimeError("No PSA directories found. The script expects to "
                           "find PSA directories named after PSA orientation "
                           "angle or in the format '<angle> deg'.")

    if 0 in psa_angles and 180 in psa_angles:
        print("Data contains 0 deg and 180 deg PSA, which is redundant. "
              "180 deg PSA will be dropped")
        rem_index = psa_angles.index(180)
        del psa_angles[rem_index]
        del psa_dirs[rem_index]

    sort_inds = np.argsort(psa_angles)
    psa_angles = [psa_angles[ind] for ind in sort_inds]
    psa_dirs = [psa_dirs[ind] for ind in sort_inds]
    num_psa = len(psa_angles)

    print("A total of {:d} PSA directories found.".format(num_psa))
    if num_psa < 8 or num_psa > 10:
        print("Unexcpected number of PSA states, it is usually from 8 to 10. "
            "Trying anyway.")

    pipo_data = None
    for ind_psa, psa_val in enumerate(psa_angles):
        psa_dir = psa_dirs[ind_psa]
        file_names = list_files_with_extension(psa_dir)
        if len(file_names) > 1:
            raise RuntimeError("More than one DAT file in {:}".format(psa_dir))
        file_name = file_names[0]

        data = load_pipo(
            file_name, chan_ind=2, num_psg_states=9, num_psa_states=1)

        if pipo_data is None:
            num_psg = data.shape[3]
            if num_psg == 19:
                print("Data contains 19 states, which is likely 0 to 180 deg "
                      "at 20 deg. Dropping the last state for a total of 18 "
                      "unique PSG states.")
                num_psg -= 1

            pipo_data = np.ndarray([*data.shape[:2], num_psa, num_psg])
            pipo_data.fill(np.nan)

        pipo_data[:, :, ind_psa, :] = data[:, :, 0, :num_psg]

        plt.clf()
        plt.imshow(np.nansum(np.nansum(pipo_data,2),2))
        plt.title("Total count image, data {:d} out of {:d}".format(
            ind_psa+1, num_psa))
        plt.draw()
        plt.pause(0.001)

    np.save('data.npy', pipo_data)
    plt.show()

    # TODO: conversion to a PIPO TIFF does not work for custom data yet
    # convert_nsmp_to_tiff(pimg_arr=pipo_data)
