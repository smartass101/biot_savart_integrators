"""Numerical routines for Biot-Savart integration of a generic current curve

The routines expect (dask-chunked) xarray inputs in order to facilitate 
memory-efficient and parallelized computation.
"""
from .integrand import biot_savart_integrand as bsintegrand


def biot_savart_integral(r, r_c, integration_dim, spatial_dim, I=1):
    # TODO make sure won't divide by coord - integration should cancel, but how accurately at edges?
    dl = r_c.differentiate(integration_dim)
    j = I*dl
    integrand = bsintegrand(r, r_c, j, spatial_dim)
    integral = integrand.integrate(integration_dim)
    return integral