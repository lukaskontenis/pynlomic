"""Convert a TIFF image to PNG.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

from lkcom.dataio import list_files_with_extension

for file_name in list_files_with_extension(ext='tif'):
    print("Reading '{:}'...".format(file_name))
    img = plt.imread(file_name)
    dtype = img.dtype
    num_rows, num_cols = np.shape(img)
    num_px = num_rows*num_cols

    min_val = np.min(img)
    max_val = np.max(img)
    num_sat_px = np.sum(img >= max_val)

    print("Image size: {:}x{:}, type: {:}, min: {:}, max: {:}".format(num_cols, num_rows, dtype, min_val, max_val))

    if max_val >= 255:
        print("Image is at risk of overexposure, {:} pixels ({:.1f}%) are satureated".format(num_sat_px, num_sat_px/num_px*100))

    out_file_name = Path(file_name).stem + '.png'

    img_out = Image.fromarray(img)
    img_out.save(out_file_name)

    img_check = plt.imread(out_file_name)
    if (img - (img_check*255).astype(dtype) != 0).any():
        print("Processed image is not the same")
    else:
        print("Image successfully converted to PNG without loss of data")
        os.remove(file_name)
