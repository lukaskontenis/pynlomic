
"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains polarimetry routines

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
from numpy import zeros, sin, cos, pi
import matplotlib.pyplot as plt

from lklib.util import isstring


def col_vec(arr):
    """Make an ndarray column vector."""
    vec = np.array(arr)
    vec.shape = (len(vec), 1)
    return vec

def get_eps():
    """Get the floating point math precision."""
    return np.finfo('float64').eps

def tensor_eq(tns1, tns2):
    """Check if two tensors are equal within the floating point precision.

    Works for vectors and matrices too.
    """
    return (np.abs(tns1 - tns2) <= get_eps()).all()

def rot_mueller_mat(mat, theta=0):
    """Get a rotated Mueller matrix."""
    mat_rot = zeros([4, 4])

    mat_rot[0, 0] = 1

    mat_rot[1, 1] = cos(2*theta)
    mat_rot[1, 2] = -sin(2*theta)
    mat_rot[2, 1] = sin(2*theta)
    mat_rot[2, 2] = cos(2*theta)

    mat_rot[3, 3] = 1

    return mat_rot @ mat @ mat_rot.transpose()

def get_mueller_mat(element, theta=0, **kwargs):
    """Get the Mueller matrix of ``element``.

    The element can be rotated by angle ``theta``. The ``PolD`` diattenuating
    polarizer transmission coefficients are specified by the ``q`` and ``r``
    kwargs.
    """
    mat = zeros([4, 4])

    element = element.lower()

    if element == "hwp":
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = -1
        mat[3, 3] = -1

    elif element == "qwp":
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 3] = 1
        mat[3, 2] = -1

    elif element in ("rtd", "retarder"):
        d = kwargs.get('d', 0) # pylint: disable=C0103
        c = cos(d) # pylint: disable=C0103
        s = sin(d) # pylint: disable=C0103

        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = c
        mat[2, 3] = s
        mat[3, 2] = -s
        mat[3, 3] = c

    elif element in ("pol", "polarizer"):
        mat[0, 0] = 1
        mat[0, 1] = 1
        mat[1, 0] = 1
        mat[1, 1] = 1
        mat = mat*0.5

    elif element in ("unity", "empty", "nop"):
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = 1
        mat[3, 3] = 1

    else:
        print("Element ''{:s}'' not defined".format(element))

    #mat = matrix(mat)
    mat = rot_mueller_mat(mat, theta)

    return mat


def get_stokes_vec(state):
    """Get the Stokes vector of ``state``."""
    svec = zeros([4, 1])

    if not isstring(state):
        # Linear polarizations. Gamma happens to be equal to LP orientation
        gamma = state / 180 * pi
        omega = 0
    else:
        state = state.lower()
        if state == "hlp":
            gamma = 0
            omega = 0
        elif state == "vlp":
            gamma = pi/2
            omega = 0
        elif state == "+45":
            gamma = pi/4
            omega = 0
        elif state == "-45":
            gamma = -pi/4
            omega = 0
        elif state == "rcp":
            gamma = 0
            omega = pi/4
        elif state == "lcp":
            gamma = 0
            omega = -pi/4
        else:
            print('State ''{:s}'' not defined'.format(state))

    svec[0] = 1
    svec[1] = cos(2*gamma) * cos(2*omega)
    svec[2] = sin(2*gamma) * cos(2*omega)
    svec[3] = sin(2*omega)

    svec.shape = (4, 1)
    #svec = matrix(svec)

    return svec


def test_pol_trans():
    """Test polarizer transmission."""
    svec_in = get_stokes_vec("HLP")

    theta_arr = np.arange(0, pi, pi/100)
    trans = np.zeros_like(theta_arr)

    for (ind, theta) in enumerate(theta_arr):
        mat_pol = get_mueller_mat("POL", theta)
        svec_out = mat_pol * svec_in
        trans[ind] = svec_out[0]

    plt.plot(theta_arr/pi*180, trans)
