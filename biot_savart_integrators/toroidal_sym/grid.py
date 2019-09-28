'''Numerical routines for calculating $\vec B (\vec r)$ of a cylindrical coil with a rectangular cross-section based on the article

L. K. URANKAR: Vector potential and magnetic field of current-carrying finite arc segment in analytical form, Part III: Exact computation for rectangular cross section,  IEEE Transactions on Magnetics ( Volume: 18, Issue: 6, Nov 1982 )

Only the forumals (3) and (4) for the $\phi$ line intergals are used for numerical integration.
'''
import numpy as np
import scipy
from scipy.integrate import romb, quad
from scipy.constants import mu_0

try:
    import numba
except ImportError:
    numba = None


def multi_integrand(phi, r, z, r_, z_):
    """Return integrand(phi) of hat {A_phi, H_r, H_z}"""
    # help vars
    gamma = z_ - z
    B = np.sqrt(r_**2 + r**2 - 2*r*r_*np.cos(phi))
    D = np.sqrt(gamma**2 + B**2)
    G = np.sqrt(gamma**2 + (r*np.sin(phi))**2)
    beta1 = (r_ - r*np.cos(phi)) / G
    beta2 = gamma / B
    beta3 = gamma * (r_ - r*np.cos(phi)) / (r * np.sin(phi) * D)
    # integrands
    A_phi = 0.5 * gamma * D + 2*gamma*r*np.cos(phi)*np.arcsinh(beta1)
    H_r = np.cos(phi) * (D + r*np.cos(phi)*np.arcsinh(beta1))
    H_z = (gamma*np.arcsinh(beta1) - r*np.cos(phi)*np.arcsinh(beta2)
           - r*np.sin(phi)*np.arctan(beta3))
    return np.array([ A_phi, H_r, H_z])


def integrad_vertex(phi, r, z, r_, z_, b1):
    """Return integrand(phi) of hat H_r, H_z for polygon vertex
    of edge v with slope 0 < |b1| < inf
    According to Part V
    """
    # help vars
    a0sq = 1 + b1**2
    a0cb = a0sq**(3/2.)         # a_0^3
    gamma = z_ - z
    r1 = r_ - b1*gamma
    D = np.sqrt(gamma**2 + r_**2 + r**2 - 2*r*r_*np.cos(phi))
    Gamma = gamma + b1*(r_ - r*np.cos(phi))
    Gsq = gamma**2 + (r*np.sin(phi))**2
    # reduces to part III thanks to sin^2 + cos^2
    B = np.sqrt((r1 - r*np.cos(phi))**2 + a0sq*(r*np.sin(phi))**2)
    beta1 = (r_ - r*np.cos(phi)) / np.sqrt(Gsq)
    beta2 = Gamma / B
    # denominator parentheses in part III, not explicit in part V
    # but needs to remain dimensionless
    beta3 = (gamma * (r_ - r*np.cos(phi)) - b1*Gsq) / (r * np.sin(phi) * D)
    # integrands
    # (1/a_0^2) in part IV probably isolated so as to reduce to part III
    # also the "full" denominators would breach dimensionality otherwise
    H_r = np.cos(phi) * (D/a0sq + r*np.cos(phi)*np.arcsinh(beta1)
                         - b1/a0cb*(r1 + b1**2*r*np.cos(phi))*np.arcsinh(beta2))
    H_z = (gamma*np.arcsinh(beta1)
           + 1/a0cb*((b1**2*r1) - (2*a0sq - 1)*r*np.cos(phi))*np.arcsinh(beta2)
           - r*np.sin(phi)*np.arctan(beta3)
           - b1**2/a0sq*D)
    return np.array([H_r, H_z])



def add_last_dims(arr, n=1):
    """Prepare array for broadcasting """
    arr = np.asarray(arr)
    return np.reshape(arr, arr.shape + (1,)*n)


def b1_edge_slopes(R, Z, axis):
    DR = np.diff(R, axis=axis)
    DZ = np.diff(Z, axis=axis)
    b1 = DR / DZ
    return b1


def B_rz_polygons_norm(r, z, R, Z, k=10, phi1=-np.pi/4, phi2=np.pi/4):
    """Calculate contribution to magnetic field at (r,z) from a train of polygons with R,Z vertices

    Parameters
    ----------
    r, z : scalar or array_like (n,)
        coordinates at which the magnetic field is calculated
    R, Z : (2, m) array_like
        2D meshgrid-like coordinates of the train (sharing opposing sides) of polygons
        each quad represents a cylindrical coil with polygonal cross-section
    k : int, optional
        Romberg integration uses 2**k + 1 samples, by default k=10
        If None, uses vectorized quad(), possibly accelerated using numba.cfunc
    phi1 : float, optional
        bottom integration limit, by default -pi/4
    phi2 : float, optional
        top integration limit, by default pi/4

    Returns
    -------
    B_norm : (2,m-1,n) ndarray
        the first dimension are the {r, z} vector components
        the next two dims are the normalized contributions of the m-th coil
        to the magnetic field at position (r,z)[n]. n will be at least 1
        To get the true magnetic field at (r,z), the result must be multiplied with
        a (m-1,1) current density grid and summed over m.

    Notes
    -----
    Romberg numerical integration is used because quad and other cannot efficiently
    integrate vector functions.
    """
    b1v = b1_edge_slopes(R, Z, axis=0)  # vertical
    b1h = b1_edge_slopes(R, Z, axis=1)  # horizontal
    # prepare for broadcasting against each other
    args_ = [add_last_dims(a, i) for (a, i) in zip((r, z, R, Z),
                                                   (1, 1, 2, 2))]
    if k:
        # sample-based Romberg integration
        x, dx = np.linspace(phi1, phi2, 2**k+1, retstep=True)
        y = multi_integrand(x, *args_)
        y[~np.isfinite(y)] = 0 # probably should cancel out anyways
        H = scipy.integrate.romb(y, dx)
    else:
        if numba:               # TODO refactor multi_integrand into separate function for quad to work
            nb_mint = numba.cfunc('float64(float64, float64, float64, float64, float64)')(multi_integrand).ctypes
            @numba.cfunc(numba.types.double(numba.types.intc, numba.types.CPointer(numba.types.double)))
            def integrand_c(n, xx_ptr):
                xx = numba.carray(xx_ptr, (n,))
                return nb_mint(xx[0], xx[1], xx[2], xx[3], xx[4])
            integrand = integrand_c.ctypes
        else:
            integrand = multi_integrand
        @np.vectorize
        def do_integrate(r, z, R, Z):
            return quad(integrand, phi1, phi2, args=(r, z, R, Z))
        H = do_integrate(*args_)
    H /= 4 * np.pi  # common for both B and A
    H = np.diff(np.diff(H, axis=1), axis=2)
    B = mu_0 * H[1:]
    A = H[0]
    return A, B

def A_B_rz_grid_norm(r, z, R, Z, k=10, phi1=0, phi2=2*np.pi):
    """Calculate contribution to magnetic field at (r,z) from current grid RxZ

    Parameters
    ----------
    r, z : scalar or array_like (n,)
        coordinates at which the magnetic field is calculated
    R, Z : (m, l) array_like
        2D meshgrid-like coordinates of a rectangular current grid,
        each quad represents a cylindrical coil with rectangular cross-section
    k : int, optional
        Romberg integration uses 2**k + 1 samples, by default k=10
        If None, uses vectorized quad(), possibly accelerated using numba.cfunc
    phi1 : float, optional
        bottom integration limit, by default 0
    phi2 : float, optional
        top integration limit, by default 2*pi

    Returns
    -------
    A_phi_norm : (m-1,l-1,n) ndarray
        toroidal component of the magnetic vector potential,
        for
    B_norm : (2,m-1,l-1,n) ndarray
        the first dimension are the {r, z} vector components
        the next three dims are the normalized contributions of each coil
        to the magnetic field at position (r,z)[n]. n will be at least 1
        To get the true magnetic field at (r,z), the result must be multiplied with
        a (m-1,l-1,1) current density grid and summed over (m,l).

    Notes
    -----
    Romberg numerical integration is used because quad and other cannot efficiently
    integrate vector functions.
    """
    # prepare for broadcasting against each other
    args_ = [add_last_dims(a, i) for (a, i) in zip((r, z, R, Z),
                                                   (1, 1, 2, 2))]
    if k:
        # sample-based Romberg integration
        x, dx = np.linspace(phi1, phi2, 2**k+1, retstep=True)
        y = multi_integrand(x, *args_)
        y[~np.isfinite(y)] = 0 # probably should cancel out anyways
        H = scipy.integrate.romb(y, dx)
    else:
        if numba:               # TODO refactor multi_integrand into separate function for quad to work
            nb_mint = numba.cfunc('float64(float64, float64, float64, float64, float64)')(multi_integrand).ctypes
            @numba.cfunc(numba.types.double(numba.types.intc, numba.types.CPointer(numba.types.double)))
            def integrand_c(n, xx_ptr):
                xx = numba.carray(xx_ptr, (n,))
                return nb_mint(xx[0], xx[1], xx[2], xx[3], xx[4])
            integrand = integrand_c.ctypes
        else:
            integrand = multi_integrand
        @np.vectorize
        def do_integrate(r, z, R, Z):
            return quad(integrand, phi1, phi2, args=(r, z, R, Z))
        H = do_integrate(*args_)
    H /= 4 * np.pi  # common for both B and A
    H = np.diff(np.diff(H, axis=1), axis=2)
    B = mu_0 * H[1:]
    A = H[0]
    return A, B


