import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynolmic",
    version="0.9.0",
    author="Lukas Kontenis",
    author_email="dse.ssd@gmail.com",
    description="A Python library for nonlinear microscopy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        'scripts/gen_img_report.py',
        'scripts/lcmicro_to_png_tiff.py',
        'scripts/make_nsmp_tiff.py',
        'scripts/make_pipo_tiff_piponator.py',
        'scripts/make_psf_figure.py',
        'scripts/tiff_to_png.py'])],
)
