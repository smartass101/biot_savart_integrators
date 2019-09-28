r'''Numerical routines for calculating $\vec B (\vec r)$ of a toroidal current with a polygonal cross-section based on the article

L. K. URANKAR: Vector potential and magnetic field of current-carrying finite arc segment in analytical form, Part V: Polygon Cross Section, IEEE Transactions on Magnetics ( Volume: 26, Issue: 3, May 1990 )

Only the formula (10b) for the $\phi$ line integrals is used for numerical integration.
'''

#TODO: possible optimizations: numba for ctype integrand, parallelize map using dask?
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.constants import mu_0

import numba
import numba.types as nut


@numba.jit(nopython=True, nogil=True)
def parametrization(phi, r, z, r_, z_, b1):
    """Precalculate parametrizations according to (5a) and (9a)"""
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
    return a0sq, a0cb, gamma, r1, D, B, Gamma, beta1, beta2, beta3


def jit_integrand_function(integrand_function):
    """Based on https://stackoverflow.com/a/49732825/4779220"""
    jitted_function = numba.jit(integrand_function, nopython=True, nogil=True)

    @numba.cfunc(nut.float64(nut.intc, nut.CPointer(nut.float64)))
    def wrapped(n, xx):
        # TODO: nicer way to not hard code number of args? `*carray()` may not expand correctly
        return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
    return LowLevelCallable(wrapped.ctypes)


@jit_integrand_function
def integrand_Hr_vertex(phi, r, z, r_, z_, b1):
    """According to (11b), l=r"""
    a0sq, a0cb, gamma, r1, D, B, Gamma, beta1, beta2, beta3 = parametrization(phi, r, z, r_, z_, b1)
    # (1/a_0^2) in part IV probably isolated so as to reduce to part III
    # also the "full" denominators would breach dimensionality otherwise
    H_r = np.cos(phi) * (D/a0sq + r*np.cos(phi)*np.arcsinh(beta1)
                         - b1/a0cb*(r1 + b1**2*r*np.cos(phi))*np.arcsinh(beta2))
    return H_r


@jit_integrand_function
def integrand_Hz_vertex(phi, r, z, r_, z_, b1):
    """According to (11b), l=z"""
    a0sq, a0cb, gamma, r1, D, B, Gamma, beta1, beta2, beta3 = parametrization(phi, r, z, r_, z_, b1)
    H_z = (gamma*np.arcsinh(beta1)
           + 1/a0cb*((b1**2*r1) - (2*a0sq - 1)*r*np.cos(phi))*np.arcsinh(beta2)
           - r*np.sin(phi)*np.arctan(beta3)
           - b1**2/a0sq*D)
    return H_z


def edge_contribution(r_1, z_1, r_2, z_2, integrand_func, phi1, phi2, Dz_min, r, z):
    Dz = z_2 - z_1
    if np.abs(Dz) < Dz_min:
        return 0
    Dr = r_2 - r_1
    b1 = Dr / Dz
    def contrib_vertex(r_, z_):
        # expected possible singularity in beta3 at 0
        contrib, err = quad(integrand_func, phi1, phi2, args=(r, z, r_, z_, b1), points=(0,))
        return -contrib         # negative both in 10b and 11b
    contrib1 = contrib_vertex(r_1, z_1)
    contrib2 = contrib_vertex(r_2, z_2)
    contrib = contrib2 - contrib1
    return contrib

def mapping_edge_contribution(args):
    edge, integrand_args = args
    return edge_contribution(*edge, *integrand_args)


def polygon_curve_integral(R_vertices, Z_vertices, integrand_args, map_func=map):
    edges_args = (((R_vertices[i-1], Z_vertices[i-1], R_vertices[i], Z_vertices[i]), integrand_args)
             for i in range(len(R_vertices)))
    contribs = map_func(mapping_edge_contribution, edges_args)
    integral = sum(contribs)
    return integral


def cw_sort_vertices(R, Z):
    Rc, Zc = (np.mean(x) for x in (R, Z))
    theta = np.arctan2(Z-Zc, R-Rc)
    ii = np.argsort(theta)
    R, Z = (x[ii] for x in (R,Z))
    return R, Z


def quad_polygon_train_gen(R, Z, circular=False):
    """Generator of (R, Z) vertices from a train of quad polygons with R,Z vertices

    Parameters
    ----------
    R, Z : (2, m) array_like
        2D meshgrid-like coordinates of the train (sharing opposing sides) of polygons
        each quad represents a cylindrical coil with polygonal cross-section
    circular: bool, optional
        if True, the last and first edge define another connecting polygon
    """
    if circular:
        R, Z = (np.hstack([x, x[:,0:1]]) for x in (R, Z))
    m = np.shape(R)[1]
    for i in range(m-1):        # right-hand orientation starting from the bottom
        yield cw_sort_vertices(*(x[:,i:i+2].ravel() for x in (R, Z)))
        

def polygon_area_shoelace(x, y=None):
    """Based on https://stackoverflow.com/a/30408825/4779220"""
    if y is None:
        x, y = x
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


_implemented_B_dirs = {
    'r': integrand_Hr_vertex,
    'z': integrand_Hz_vertex,
}


def _mapping_polygons(args):
    poly, integrand_args, map_func = args
    return polygon_curve_integral(poly[0], poly[1], integrand_args, map_func)

def _mapping_loc_H(args):
    polygons, integrand_args, map_funcs = args
    return list(map_funcs[1](_mapping_polygons, ((poly, integrand_args, map_funcs[2])
                                            for poly in polygons)))



def B_norm_contribs(direction, r, z, polygons, phi1=-np.pi/4, phi2=np.pi/4, Dz_min=1e-6, map_funcs=(map,)*3):
    """Calculate contribution to magnetic field at (r,z) from circular coils with polygon cross-sections

    Parameters
    ----------
    direction : str
       one of {}
    r, z : scalar or array_like (n,)
        coordinates at which the magnetic field is calculated
    polygons : sequence, (m,)
        (circular) sequence of [(R_i, Z_i)] polygon vertices in edge order
        the vertex train should have right-hand orientation in the R,Z plane
        can be generated by :func:`quad_polygon_train_gen`
    phi1 : float, optional
        bottom integration limit, by default -pi/4
        toroidal angle relative to phi=0 at (r,z) coordinates
    phi2 : float, optional
        top integration limit, by default pi/4
    eps : float
        minimum $\\Delta Z$ separation on edge to be calculated,
        for $\\Delta Z = 0$ the edge contribution vanishes and may be numerically unstable
    map_funcs : sequence of map-like funcs, (3,)
        map-like functions (could be e.g. :meth:`multiprocessing.Pool.map``),
        each element corresponds to a different looping level:
        0: looping over (r, z) coordinates (if array_like)
        1: looping over polygons
        2: looping over polygon edges

    Returns
    -------
    B_norm : (n, m) ndarray
        the first dimension is the normalized contributions of the n-th coil
        to the magnetic field at position (r,z)[n]. n will be at least 1
        To get the true magnetic field at (r,z), the result must be multiplied with
        a (1,m-1) current density grid and summed over m, i.e. a dot product
        with a (m-1,1) column vector
    """
    try:
        integrand_func = _implemented_B_dirs[direction]
    except KeyError:
        raise ValueError('unknown B direction "{}", only {:!r} supported'.format(
            direction, set(_implemented_B_dirs.keys())))
    integrand_args = (integrand_func, phi1, phi2, Dz_min)
    loc_dim = np.ndim(r)
    if loc_dim == 0:
        H = _mapping_loc_H((polygons, integrand_args + (r, z), map_funcs))
    elif loc_dim == 1:
        H = map_funcs[0](_mapping_loc_H, ((polygons, integrand_args + loc, map_funcs)
                                          for loc in zip(r, z)))
    B = mu_0 / (4*np.pi) * np.asarray(list(H))
    return B

