import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lcmicro",
    version="0.0.1",
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
        'lklib'
    ],
    python_requires='>=3.6',
)