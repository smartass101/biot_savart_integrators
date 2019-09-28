from biot_savart_integrators.generic import curve as generic_curve

import numpy as np
import xarray as xr
from scipy.constants import pi, mu_0

import pytest


@pytest.fixture
def setup():
    I = 1.15766
    a = 6.789  # circle radius
     # high phi resolution to fulfill tolerance
    phi = xr.DataArray(np.linspace(-np.pi, np.pi, int(1e6)), dims=['phi'], name='phi')
    r0 = xr.DataArray([-5.7, 8.9, 2.1], coords=[('s', list('xyz'))], name='r0')
    r_c = xr.concat([a*np.cos(phi), a*np.sin(phi), xr.zeros_like(phi)], r0.s) + r0
    return I, a, r0, r_c


def test_circle_winding(setup):
    """Compare numerical integration with analytical result"""
    I, a, r0, r_c = setup
    Bz_analytic = mu_0*I/(2*a)
   
    B_calc = generic_curve.biot_savart_integral(r0, r_c, integration_dim='phi',
                                       spatial_dim='s', I=I)
    np.testing.assert_allclose(B_calc.sel(s=['x', 'y']), [0,0])
    np.testing.assert_allclose(B_calc.sel(s='z'), Bz_analytic)


def test_circle_winding_profile(setup):
    I, a, r0, r_c = setup

    z = xr.DataArray(np.zeros((50, 3)), dims=['z', 's'], coords={'s': r0.s})
    z.loc[{'s': 'z'}] = np.linspace(0, 3.6, 50)
    r = r0 + z
    
    Bz_analytic = mu_0*I/2*a**2/(z.sel(s='z')**2+a**2)**(3/2.)
    
    B_calc = generic_curve.biot_savart_integral(r.chunk({'z': 1}), r_c, integration_dim='phi',
                                       spatial_dim='s', I=I).compute()
    np.testing.assert_allclose(B_calc.sel(s='z'), Bz_analytic)

