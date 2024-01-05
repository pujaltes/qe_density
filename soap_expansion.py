import numpy as np
import numba as nb
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import tplquad, nquad
from get_rx import get_rx


@nb.njit
def symm_real_sqrtm(A):
    """Compute the square root of a SYMMETRIC REAL matrix."""
    # https://stackoverflow.com/questions/61262772/is-there-any-way-to-get-a-sqrt-of-a-matrix-in-numpy-not-element-wise-but-as-a
    evalues, evectors = np.linalg.eigh(A)
    assert (evalues >= 0).all()
    sqrt_matrix = evectors * np.sqrt(evalues) @ evectors.T
    return sqrt_matrix


@nb.njit
def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def get_basis_poly(r_cut, n_max):
    """Used to calculate discrete vectors for the polynomial basis functions.

    Args:
        r_cut(float): Radial cutoff.
        n_max(int): Number of polynomial radial bases.

    Returns:
        (np.ndarray, np.ndarray): Tuple containing the evaluation points in
        radial direction as the first item, and the corresponding
        orthonormalized polynomial radial basis set as the second item.
    """
    # EQ.
    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    S = np.zeros((n_max, n_max), dtype=np.float64)
    # NOTE: maybe optimize this loop if it is too slow
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            S[i - 1, j - 1] = (2 * (r_cut) ** (7 + i + j)) / ((5 + i + j) * (6 + i + j) * (7 + i + j))

    # Get the beta factors that orthonormalize the set with Löwdin
    # orthonormalization
    betas = symm_real_sqrtm(np.linalg.inv(S))

    def rbf_polynomial(r, n, l):
        poly = 0
        for k in range(1, n_max + 1):
            poly += betas[n, k - 1] * (r_cut - np.clip(r, 0, r_cut)) ** (k + 2)
        return poly

    return rbf_polynomial


def integral(grid_rho, grid_coords, n, l, m, rbf_function, r_cut):
    # Integration limits for radius
    r1 = 0.0
    r2 = r_cut + 5

    # Integration limits for theta
    t1 = 0
    t2 = np.pi

    # Integration limits for phi
    p1 = 0
    p2 = 2 * np.pi

    # Get the grid spacing
    x_idx = grid_coords[:, 0, 0][:, 0]
    y_idx = grid_coords[0, :, 0][:, 1]
    z_idx = grid_coords[0, 0, :][:, 2]

    def soap_coeff(phi, theta, r):
        print('hey')
        # Regular spherical harmonic, notice the abs(m)
        # needed for constructing the real form
        ylm_comp = scipy.special.sph_harm(np.abs(m), l, phi, theta)  # NOTE: scipy swaps phi and theta

        # Construct real (tesseral) spherical harmonics for
        # easier integration without having to worry about
        # the imaginary part. The real spherical harmonics
        # span the same space, but are just computationally
        # easier.
        ylm_real = np.real(ylm_comp)
        ylm_imag = np.imag(ylm_comp)
        if m < 0:
            ylm = np.sqrt(2) * (-1) ** m * ylm_imag
        elif m == 0:
            ylm = ylm_comp
        else:
            ylm = np.sqrt(2) * (-1) ** m * ylm_real

        # Atomic density
        interpolator = RegularGridInterpolator((x_idx, y_idx, z_idx), grid_rho, bounds_error=False, fill_value=0)
        x, y, z = spherical_to_cartesian(r, theta, phi)
        rho = interpolator((x, y, z))

        # Jacobian
        jacobian = np.sin(theta) * r**2

        return rbf_function(r, n, l) * ylm * rho * jacobian
    # NOTE: these lambda function are unnecessary, we can pass a float directly
    # cnlm = tplquad(
    #     soap_coeff,
    #     r1,
    #     r2,
    #     lambda r: t1,
    #     lambda r: t2,
    #     lambda r, theta: p1,
    #     lambda r, theta: p2,
    #     epsabs=1e-6,
    #     epsrel=1e-4,
    # )
    cnlm = nquad(
        soap_coeff,
        [[r1, r2], [t1, t2], [p1, p2]],
        opts={"points": grid_coords.reshape(180 ** 3, 3), "epsabs": 1e-6, "epsrel": 1e-4},
    )
    integral, error = cnlm

    return integral

integral(dens.rho, coords - 14.8, 2, 2, 2, get_basis_poly(2, 3), 2)


m=2
l=2

def soap_coeff(phi, theta, r):
    #print('hey')
    # Regular spherical harmonic, notice the abs(m)
    # needed for constructing the real form
    ylm_comp = scipy.special.sph_harm(np.abs(m), l, phi, theta)  # NOTE: scipy swaps phi and theta

    # Construct real (tesseral) spherical harmonics for
    # easier integration without having to worry about
    # the imaginary part. The real spherical harmonics
    # span the same space, but are just computationally
    # easier.
    ylm_real = np.real(ylm_comp)
    ylm_imag = np.imag(ylm_comp)
    if m < 0:
        ylm = np.sqrt(2) * (-1) ** m * ylm_imag
    elif m == 0:
        ylm = ylm_comp
    else:
        ylm = np.sqrt(2) * (-1) ** m * ylm_real

for i in range(int(1e6)):
    soap_coeff(0, 0, 0) 

#