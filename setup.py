import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lcmicro",
    version="0.5.5",
    author="Lukas Kontenis",
    author_email="dse.ssd@gmail.com",
    description="A Python library for nonlinear microscopy and polarimetry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/lukaskontenis/lcmicro/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'lklib>=0.0.15'
    ],
    python_requires='>=3.6',
    data_files=[
        ('scripts', [
        'scripts/calib_laser_power.py',
        'scripts/fit_pipo_1point.py',
        'scripts/fit_pipo_1point_zcq.py',
        'scripts/fit_pipo_img.py',
        'scripts/gen_img_report.py',
        'scripts/gen_pol_state_sequence.py',
        'scripts/lcmicro_to_png_tiff.py',
        'scripts/make_nsmp_tiff.py',
        'scripts/make_pipo_tiff_piponator.py',
        'scripts/make_psf_figure.py',
        'scripts/pipo_check_c6v.py',
        'scripts/plot_pipo_fit.py',
        'scripts/plot_piponator_fit.py',
        'scripts/show_pipo.py',
        'scripts/sim_collagen_anim.py',
        'scripts/sim_pipo.py',
        'scripts/sim_pipo_collagen.bat',
        'scripts/sim_pipo_collagen_hr.bat',
        'scripts/sim_pipo_rtt.py',
        'scripts/sim_pipo_zcq.bat',
        'scripts/sim_pipo_zcq_hr.bat',
        'scripts/sim_zcq_pipo_anim.py',
        'scripts/verify_pol_state_sequence.py']),
        ('tests', [
        'tests/pipo_8x8_pol_states.dat',
        'tests/test_polarimetry_fit.py',
        'tests/test_polarimetry_lin.py',
        'tests/test_polarimetry_nl.py',
        'tests/test_polarimetry_plot.py'])],
)
