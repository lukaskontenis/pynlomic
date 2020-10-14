
""" lcmicro - a Python library for nonlinear microscopy and polarimetry.

This module contains polarimetry routines

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
from numpy import zeros, sin, cos, pi
import matplotlib.pyplot as plt

from lklib.util import isstring
from lklib.plot import add_x_marker, add_y_marker


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
        d = kwargs.get('d', 0)  # pylint: disable=C0103
        c = cos(d)  # pylint: disable=C0103
        s = sin(d)  # pylint: disable=C0103

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
    return svec


def get_waveplate_thickness(
        plate_type='hwp', wavl=None, biref=0.0092, worder=0):
    """Get waveplate thickness given its order and birefringence.

    Default birefringence is 0.0092 for a quartz waveplate in green. Thickess
    is returned in the same units as wavelength.
    """
    if plate_type is 'hwp':
        fac = 1
    elif plate_type is 'qwp':
        fac = 0.5
    else:
        print("Unsupported plate type " + plate_type)
        return None
    return (2*worder + 1)*wavl*fac / (2*biref)


def get_waveplate_retardation(wavl=None, biref=0.0092, thickness=None):
    """Get waveplate retardation given its birefringence and thickness.

    Default birefringence is 0.0092 for a quartz waveplate in green.
    Retartadtion is returned in waves.
    """
    rtrd = (2*biref*thickness/wavl)  # in terms of pi
    rtrd = rtrd - 2*np.floor(rtrd/2)  # Subtract full waves, in terms of pi
    rtrd = rtrd/2  # Retardance in terms of waves
    return rtrd


def plot_waveplate_response(
        plate_type='hwp', rtrd=None, title_str=None, finalize_figure=True):
    """Plot intensity transited through a rotatating waveplate and a polarizer.

    """
    if rtrd is None:
        if plate_type is 'hwp':
            rtrd = 0.5
        elif plate_type is 'qwp':
            rtrd = 0.25
        else:
            print("Unsupported plate type " + plate_type)
            return None

    in_svec = get_stokes_vec('hlp')
    pol_hwp = get_mueller_mat('pol')

    theta_arr = np.linspace(0, 2*np.pi, 500)
    det_ampl = np.empty_like(theta_arr)

    for ind, theta in enumerate(theta_arr):
        hwp_mat = get_mueller_mat('rtd', theta=theta, d=rtrd*2*np.pi)
        out_svec = pol_hwp.dot(hwp_mat.dot(in_svec))
        det_ampl[ind] = out_svec[0]

    plt.plot(theta_arr/np.pi*180, det_ampl)

    if finalize_figure:
        add_y_marker(0)
        add_y_marker(1)
        plt.xlim([0, 360])
        plt.grid('on')
        plt.xticks(np.arange(0, 361, 45))
        if title_str is None:
            title_str = 'Rotating {:s} response'.format(plate_type.upper())
        plt.title(title_str)
        plt.xlabel('{:s} orientation, deg'.format(plate_type.upper()))
        plt.ylabel('Transmitted power, a.u.')


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
