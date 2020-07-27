"""lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains image tiling routines

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

from lklib.util import isnone, handle_general_exception
from lklib.fileread import list_files_with_extension, read_bin_file, \
     rem_extension
from lklib.cfgparse import read_cfg
from lklib.image import normalize, tile_img, tile_img_blind, save_img

from lcmicro.proc import proc_img, get_scan_artefact_sz
from lcmicro.cfgparse import get_idx_mask, get_stage_pos, get_stage_xyz_pos, \
    get_tiling_cfg


def get_tile_stage_xyz_pos(config=None, file_name=None,
                           force_reconstruct=False):
    """Get the XYZ sample stage positions for individual tiles."""
    if isnone(config):
        config = read_cfg(file_name)

    mask = get_idx_mask(config, 2)
    pos = np.ndarray([len(mask), 3])

    for (ind, index) in enumerate(mask):
        pos[ind, :] = get_stage_xyz_pos(config, index=index)

    if(force_reconstruct or (pos == 0).all()):
        print("No tiling coordinates in index. " +
              "Reconstructing from tiling config.")

        [from_x, to_x, from_y, _, step] = get_tiling_cfg(config)
        num_x_tiles = int(np.ceil(np.abs(to_x - from_x))/step) + 1

        z_pos = get_stage_pos(config, "Z")

        for ind in range(0, len(mask)):
            pos[ind, 0] = from_x + np.mod(ind, num_x_tiles) * step
            pos[ind, 1] = from_y + np.floor(ind/num_x_tiles) * step
            pos[ind, 2] = z_pos

    return pos


def get_tile_ij_idx(**kwargs):
    """Get tile ij indices from tile stage center positions."""
    bad_ij = False
    pos = get_tile_stage_xyz_pos(**kwargs)
    step = pos[1, 0] - pos[0, 0]

    # Pylint E1136 is a known issue - pylint/issues/3139.

    ij = np.ndarray([pos.shape[0], 2], dtype=np.int32)  # pylint: disable=E1136

    # This is probably swapped axes since X is ind=0 in pos and row is ind=0
    # in ij. But since such a swap is required anyway, it all works out.
    for ind in range(0, pos.shape[0]):  # pylint: disable=E1136
        ij[ind, 0] = int(np.round((pos[ind, 0] - pos[0, 0])/step))
        ij[ind, 1] = int(np.round((pos[ind, 1] - pos[0, 1])/step))

    # Reverse ij order to correspond to physical axis orientation
    ij[:, 0] = ij[:, 0].max() - ij[:, 0]
    ij[:, 1] = ij[:, 1].max() - ij[:, 1]

    # Verify that the IJ indices are positive
    if(ij < 0).any():
        bad_ij = True
        print("Some ij indices are negative!")

    # Verify that there are no duplicate ij indices
    bad_ind = 0
    for ind1 in range(0, ij.shape[0]):  # pylint: disable=E1136
        for ind2 in range(ind1+1, ij.shape[0]):  # pylint: disable=E1136
            if (ij[ind1, :] == ij[ind2, :]).all():
                bad_ind += 1

    if bad_ind:
        bad_ij = True
        print("There are %d duplicate ij indices!" % bad_ind)

    if bad_ij:
        print("ij indices don't make sense.")
        if kwargs.get('force_return_ij'):
            return ij
        else:
            return None
    else:
        return ij


def get_tiling_grid_sz(**kwargs):
    """Get tiling grid size."""
    ij = get_tile_ij_idx(**kwargs)
    return [max(ij[:, 0]), max(ij[:, 1])]


def show_raw_tiled_img(file_name=None, data=None, rng=None, save_images=True):
    """Show tiled images arranged in a mosaic without including overlap.

    Show tiles with correct tiling geometry but ignoring tiling step size and
    overlap.
    """
    try:
        if isnone(data):
            data = read_bin_file(file_name)

        [data, mask, ij] = get_tiling_data(data=data, file_name=file_name)

        mosaic = show_mosaic_img(data=data, mask=mask, ij=ij, rng=rng)

        if save_images:
            save_img(
                mosaic, ImageName=rem_extension(file_name) + "RawTiled",
                cmap="viridis", bake_cmap=True)
    except:  # pylint: disable=W0702
        handle_general_exception("Could not generate raw tiled image")


def tile_imgs_sbs(file_names=None, path=None, sort_by_y=False):
    """Tile images side-by-side.

    The input files can be provided as a list of file
    names or a path can be given where all .dat files reside.
    """
    if isnone(file_names) and isnone(path):
        print("Either file_names or path should be provided")
        return None

    if isnone(file_names):
        file_names = list_files_with_extension(path=path, ext="dat")

    if sort_by_y:
        pos = get_stage_xyz_pos(file_name=file_names)

        # Get image sort order by Y position
        tile_inds = np.argsort(pos[:, 1])

        file_names_sorted = [file_names[i] for i in tile_inds]
        file_names = file_names_sorted

    num_img = len(file_names)

    # Tile images
    img_comb = normalize(proc_img(file_names[0])[0].astype(np.uint8))
    for ind in range(1, num_img):
        img1 = normalize(proc_img(file_names[ind])[0].astype(np.uint8))
        img_comb = tile_img(img_comb, img1)

    return img_comb


def get_tiling_data(data=None, file_name=None):
    """Get tiling data, mask and ij indices."""
    if isnone(data):
        print("Reading data...")
        data = read_bin_file(file_name)

    config = read_cfg(file_name)
    mask = get_idx_mask(config, 2)

    ij = get_tile_ij_idx(config=config)

    return [data, mask, ij]


def tile_images(
        data=None, file_name=None, img_sz=None, step_sz=None, rng=None,
        save_images=True, rng_override=None):
    """Arrange images into a tiled mosaic.

    Tiling is done blindly accounting for tile overlap using tile_img_blind
    from lklib.image.

    Arguments:
        file_name - file name of data to tile
        img_sz - image size as [Y, X] in um
        step_sz - tiling grid spacing as [Y, X] in um
        rng - image value display mapping range as [min, max]
    """
    if isnone(img_sz):
        img_sz = [780, 780]
    if isnone(step_sz):
        step_sz = [510, 525]

    print("Getting tiling data...")
    [data, mask, ij] = get_tiling_data(data=data, file_name=file_name)

    print("Tiling...")
    img_tiled = tile_img_blind(
        data=data, mask=mask, ij=ij, img_sz=img_sz, step_sz=step_sz, rng=rng,
        rng_override=rng_override)

    # crop the righ-hand side scan artefact which is not removed during blind
    # tiling
    crop_px = get_scan_artefact_sz(file_name=file_name)
    img_tiled = img_tiled[:, :-crop_px]  # pylint: disable=E1130

    print("Displaying...")
    plt.clf()
    plt.imshow(img_tiled)

    if save_images:
        print("Writing tiled images...")
        save_img(
            img_tiled, ImageName=rem_extension(file_name) + "Tiled",
            cmap=["viridis", "Greys"], bake_cmap=True)

    print("All done.")
