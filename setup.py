import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biot-savart-integrators",
    version="0.0.1",
    author="Ondrej Grover",
    author_email="grover@ipp.cas.cz",
    description="Numerical integration routines for the Biot-Savart law",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'numba', 'xarray', 'dask'],
)