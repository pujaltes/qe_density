import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.interpolate import RegularGridInterpolator
from grid.utils import generate_real_spherical_harmonics as grsh
from scipy.linalg import lstsq
from scipy.special import gamma


def get_dens_interpolator(grid_coords, f_values, method="linear", fill_value=None):
    bounds_error = False
    if fill_value is None:
        bounds_error = True        
    # method options: nearest, linear with new spline options: slinear, cubic and quintic
    # TODO: implement better interpolation methods such as rbf
    grid_points = (grid_coords[:, 0, 0][:, 0], grid_coords[0, :, 0][:, 1], grid_coords[0, 0, :][:, 2])
    if method in ["slinear", "cubic", "quintic"]:
        # TODO chunked interpolation
        raise NotImplementedError("Only nearest and linear interpolation are currently supported.")
    interp = RegularGridInterpolator(
        grid_points, f_values, method=method, fill_value=fill_value, bounds_error=bounds_error
    )

    return interp


def gen_sph_orders(lmax):
    """
    Generate the spherical harmonic orders (l, m) for a given lmax in the same order
    as the grid.utils import RealSphericalHarmonics
    Note: for SHs l is degree and m is order
    """
    orders = np.zeros(((lmax + 1) ** 2, 2), dtype=np.int64)
    for l in range(lmax + 1):
        orders[l**2 : (l + 1) ** 2] = l, 0
        for m in range(1, l + 1):
            orders[l**2 + 2 * m - 1] = l, m
            orders[l**2 + 2 * m] = l, -m
    return orders


@nb.njit
def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


@nb.njit
def cartesian_to_spherical(x, y, z):
    """Convert cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


@nb.njit
def cartesian_to_spherical_angles(x, y, z, r):
    """Convert cartesian coordinates to spherical coordinates."""
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi


def radial_gto(alphas, betas, rx):
    """
    Computes RBFs for GTO basis given precomputed alphas and betas (coeffs)
    at the given rx radial coordinates.

    Use the SOAP.get_basis_gto() method to get the alphas and betas.
    """
    rx = np.expand_dims(rx, axis=(0, 1))
    l_max = alphas.shape[0] - 1
    phi = rx ** np.arange(0, l_max + 1)[:, None, None] * np.exp(-alphas[..., None] * rx**2)
    g = np.sum(betas[..., None] * phi[:, :, None], axis=1)

    return g


def get_basis_gto(r_cut, n_max, l_max, rx):
    """Used to calculate the alpha and beta prefactors for the gto-radial"""
    # These are the values for where the different basis functions should decay
    # to: evenly space between 1 angstrom and r_cut.
    a = np.linspace(1, r_cut, n_max)
    threshold = 1e-3  # This is the fixed gaussian decay threshold

    alphas_full = np.zeros((l_max + 1, n_max))
    betas_full = np.zeros((l_max + 1, n_max, n_max))

    for l in range(0, l_max + 1):
        # The alphas are calculated so that the GTOs will decay to the set
        # threshold value at their respective cutoffs
        alphas = -np.log(threshold / np.power(a, l)) / a**2

        # Calculate the overlap matrix
        m = np.zeros((alphas.shape[0], alphas.shape[0]))
        m[:, :] = alphas
        m = m + m.transpose()
        S = 0.5 * gamma(l + 3.0 / 2.0) * m ** (-l - 3.0 / 2.0)

        # Get the beta factors that orthonormalize the set with Löwdin
        # orthonormalization
        betas = sqrtm(np.linalg.inv(S))

        # If the result is complex, the calculation is currently halted.
        if betas.dtype == np.complex128:
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (n_max) or increasing the radial "
                "cutoff (r_cut) is advised."
            )

        alphas_full[l, :] = alphas
        betas_full[l, :, :] = betas

    gss = radial_gto(alphas_full, betas_full, rx)

    return gss


def get_basis_poly(r_cut, n_max, rx, cut_pad=0):
    """Used to calculate discrete vectors for the polynomial basis functions.

    Args:
        r_cut(float): Radial cutoff.
        n_max(int): Number of polynomial radial bases.

    Returns:
        (np.ndarray, np.ndarray): Tuple containing the evaluation points in
        radial direction as the first item, and the corresponding
        orthonormalized polynomial radial basis set as the second item.
    """
    r_cut += cut_pad
    if r_cut < 0:
        raise ValueError("Radial cutoff (r_cut) must be positive.")
    # if rx.max() > r_cut or rx.min() < 0:
    #     raise ValueError("Evaluation points (rx) must be within the radial cutoff.")
    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    S = np.zeros((n_max, n_max), dtype=np.float64)
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            S[i - 1, j - 1] = (2 * (r_cut) ** (7 + i + j)) / ((5 + i + j) * (6 + i + j) * (7 + i + j))

    # Get the beta factors that orthonormalize the set with Löwdin
    # orthonormalization
    betas = sqrtm(np.linalg.inv(S))

    # If the result is complex, the calculation is currently halted.
    if betas.dtype == np.complex128:
        raise ValueError(
            "Could not calculate normalization factors for the radial "
            "basis in the domain of real numbers. Lowering the number of "
            "radial basis functions (n_max) or increasing the radial "
            "cutoff (r_cut) is advised."
        )

    # Calculate the value of the orthonormalized polynomial basis at the rx
    # values
    fs = np.zeros([n_max, len(rx)])
    for n in range(1, n_max + 1):
        fs[n - 1, :] = (r_cut - rx) ** (n + 2)

    gss = np.dot(betas, fs)

    return gss


def lsq_sph_coeffs(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    """
    Least squares fit of spherical harmonics to electron density.

    Parameters
    ----------
    rho : np.array (size (n_x, n_y, n_z))
        Electron density data.
    atoms_positions : np.array
        Positions of the atoms.
    coords : np.array
        Coordinates of the grid points corresponding to the electron density.
    r_cut : float
        Cutoff radius.
    l_max : int
        Maximum azimuthal quantum number.
    n_max : int
        Maximum radial quantum number.

    Returns
    -------
    tuple
        A tuple containing:
        - coeffs : array
            Coefficients obtained from the least squares fit.
        - f : list
            List of factors used in the fit for each atom.
        - masks : array
            Boolean masks indicating regions within the cutoff radius.

    Notes
    -----
    Additional information or notes about the function.
    """
    centred_coords = coords - atoms_positions[:, None, None, None]
    radii = np.linalg.norm(centred_coords, axis=-1)
    masks = radii < r_cut
    coeffs = np.zeros((len(atoms_positions), n_max * (l_max + 1) ** 2))
    f = []
    for i, ap in enumerate(atoms_positions):
        # print(i)
        mask = masks[i]
        r = radii[i][mask]
        c_coords = centred_coords[i][mask]

        rho_vals = rho[mask]
        theta, phi = cartesian_to_spherical_angles(*c_coords.T, r)
        sh_vals = grsh(l_max, theta, phi)
        poly_basis = get_basis_poly(r_cut=r_cut, n_max=n_max, rx=r, cut_pad=cut_pad)
        factors = np.einsum("ik,jk->ijk", poly_basis, sh_vals).reshape(n_max * (l_max + 1) ** 2, -1)
        f.append(factors)
        # compute coefficients using least squares
        coeffs[i], residuals, rank, s = lstsq(factors.T, rho_vals)
    return coeffs, f, masks


def lsq_sph_coeffs_mod(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    centred_coords = coords - atoms_positions[:, None, None, None]
    radii = np.linalg.norm(centred_coords, axis=-1)
    masks = radii < r_cut
    coeffs = np.zeros((len(atoms_positions), n_max * (l_max + 1) ** 2))
    f = []
    for i, ap in enumerate(atoms_positions):
        print(i)
        mask = masks[i]
        r = radii[i][mask]
        c_coords = centred_coords[i][mask]

        rho_vals = rho[mask]
        theta, phi = cartesian_to_spherical_angles(*c_coords.T, r)
        sh_vals = grsh(l_max, theta, phi)
        poly_basis = get_basis_poly(r_cut=r_cut, n_max=n_max, rx=r, cut_pad=cut_pad)
        factors = np.einsum("ik,jk->ijk", poly_basis, sh_vals).reshape(n_max * (l_max + 1) ** 2, -1)
        f.append(factors)
        # compute coefficients using least squares
        coeffs[i], residuals, rank, s = lstsq(factors.T, rho_vals)
    return coeffs, f, masks


def reconstructor(f, mask, coeffs, compute_overlaps=True):
    """
    Reconstructs density from factors, masks, and coefficients.

    Parameters
    ----------
    f : list
        List of factors used in the fit for each atom.
    mask : array-like
        Boolean masks indicating regions within the cutoff radius.
    coeffs : array-like
        Coefficients obtained from the least squares fit.
    compute_overlaps : bool, optional
        Flag indicating whether to compute overlaps. Default is True.

    Returns
    -------
    array
        Reconstructed density.

    Notes
    -----
    Additional information or notes about the function.
    """
    new_rho = np.zeros(mask.shape[1:])
    for i, m in enumerate(mask):
        new_rho[m] += f[i].T @ coeffs[i]
    if not compute_overlaps:
        return new_rho
    overlaps = np.sum(mask, axis=0)
    overlaps[overlaps == 0] = 1
    return new_rho / overlaps


def example_basis_poly(r_cut=3, n_max=4, plot=False):
    """Get and plot simple example of polynomial basis functions."""
    rx = np.arange(10000) / 10000 * r_cut
    #    rx = (get_rx() + 1) / 2 * r_cut
    gss = get_basis_poly(r_cut, n_max, rx)
    if plot:
        plt.plot(rx, gss.T)
        plt.xlim(0, r_cut)
        plt.ylim(-1, 3)
        for n in range(n_max):
            plt.plot(rx, gss[n], label=f"n={n}")
        plt.legend()
        plt.show()
    return rx, gss


@nb.njit
def symm_real_sqrtm(A):
    """
    Compute the square root of a SYMMETRIC REAL matrix.

    Not really needed as our use is non speed critical and we can just use scipy.linalg.sqrtm
    but it may come in handy later.
    """
    # https://stackoverflow.com/questions/61262772/is-there-any-way-to-get-a-sqrt-of-a-matrix-in-numpy-not-element-wise-but-as-a
    evalues, evectors = np.linalg.eigh(A)
    assert (evalues >= 0).all()
    sqrt_matrix = evectors * np.sqrt(evalues) @ evectors.T
    return sqrt_matrix


"""
rx, gss = example_basis_poly(plot=True)

np.dot(gss[3], gss[2])
np.dot(gss[3] * rx, gss[2] * rx)


x = np.arange(100) / 100 * 3
gss = get_basis_poly(7, 4, x)
plt.plot(x, gss.T)
plt.xlim(0, 3)
plt.ylim(-1, 3)


from scipy.linalg import sqrtm

gss2 = get_basis_poly(3, 4, rx)
plt.plot(rx, gss2.T)


rx2, gss2 = get_basis_poly(3, 4, x)
plt.plot(rx2, gss2.T)


r_cut = 3
n_max = 4
l_max = 2
r = np.arange(100) / 100 * r_cut


rx2 = np.arange(10000) / 10000 * r_cut

alphas, betas = soap.get_basis_gto(r_cut, n_max, l_max)
gss_gto = radial_gto(alphas, betas, rx2)
plt.plot(rx, gss_gto[0].T * rx2[:, None])

np.dot(gss_gto[0][0], gss_gto[0][1])
np.dot(gss_gto[0][0] * rx2, gss_gto[0][1] * rx2)

plt.plot(r[0, 0], g[0].T)
plt.plot(r[0, 0], g[1].T)
plt.plot(r[0, 0], g[2].T)


# This should result in correct polynomial basis functions but there is an error in the soap code
from dscribe.descriptors import SOAP

soap = SOAP(
    species=["H"],
    r_cut=3,
    n_max=4,
    l_max=2,
    rbf="polynomial",
)
rx, gss = soap.get_basis_poly(3, 4)
plt.plot(rx, rx[:, None] * gss.T)

plt.xlim(0, 3)
"""

# def get_basis_poly_incorrect(r_cut, n_max, rx):
#     """Used to calculate discrete vectors for the polynomial basis functions.

#     Args:
#         r_cut(float): Radial cutoff.
#         n_max(int): Number of polynomial radial bases.
#         rx(np.ndarray): Array of radial coordinates (r) to evaluate the basis on

#     Returns:
#         gss(np.ndarray): Array of shape (n_max, len(rx)) containing the orthonormalized polynomial
#             radial basis set evaluated at the rx values.
#     """
#     if r_cut < 0:
#         raise ValueError("Radial cutoff (r_cut) must be positive.")
#     if rx.max() > r_cut or rx.min() < 0:
#         raise ValueError("Evaluation points (rx) must be within the radial cutoff.")
#     # Calculate the overlap of the different polynomial functions in a
#     # matrix S. These overlaps defined through the dot product over the
#     # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
#     # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
#     # the basis orthonormal are given by B=S^{-1/2}
#     # for more info see: https://doi.org/10.1103/PhysRevB.87.184115 (on representing chemical environments)
#     S = np.zeros((n_max, n_max), dtype=np.float64)
#     norm_factors = np.zeros(n_max, dtype=np.float64)
#     for i in range(1, n_max + 1):
#         norm_factors[i - 1] = np.sqrt(r_cut ** (2 * i + 5) / (2 * i + 5))
#         for j in range(1, n_max + 1):
#             S[i - 1, j - 1] = np.sqrt((5 + 2 * j) * (5 + 2 * i)) / (5 + i + j)

#     # Get the beta factors that orthonormalize the set with Löwdin
#     # orthonormalization
#     betas = sqrtm(np.linalg.inv(S)) / norm_factors[None, :]

#     # If the result is complex, the calculation is currently halted.
#     if betas.dtype == np.complex128:
#         raise ValueError(
#             "Could not calculate normalization factors for the radial "
#             "basis in the domain of real numbers. Lowering the number of "
#             "radial basis functions (nmax) or increasing the radial "
#             "cutoff (rcut) is advised."
#         )
#     # Calculate the value of the orthonormalized polynomial basis at the rx values
#     fs = np.zeros([n_max, len(rx)])
#     for n in range(1, n_max + 1):
#         fs[n - 1, :] = (r_cut - rx) ** (n + 2)

#     gss = np.dot(betas, fs)

#     return gss
