"""Read and plot PIPONATOR fit results.

Read PIPONATOR CAK and SCK fit results, convert them to ImgFitData and plot
using the lcmicro plotter.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

from lcmicro.polarimetry.report import plot_piponator_fit

plot_piponator_fit()
