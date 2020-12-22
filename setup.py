import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lcmicro",
    version="0.1.2",
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
        'lklib>=0.0.13'
    ],
    python_requires='>=3.6',
        data_files=[
        ('scripts', [
        'scripts/calib_laser_power.py',
        'scripts/gen_img_report.py',
        'scripts/lcmicro_to_png_tiff.py',
        'scripts/make_psf_figure.py',
        'scripts/sim_collagen_anim.py',
        'scripts/sim_zcq_pipo_anim.py']),
        ('tests', [
        'tests/pipo_8x8_pol_states.dat',
        'tests/test_polarimetry_lin.py',
        'tests/test_polarimetry_nl.py',
        'tests/test_polarimetry_plot.py'])],
)
