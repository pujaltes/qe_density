import numpy as np
import numba as nb
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


@nb.njit
def cartesian_to_spherical(x, y, z):
    """Convert cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


@nb.njit
def get_basis_poly_betas(r_cut, n_max):
    """Used to calculate discrete beta (values) vectors for the polynomial basis functions.

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
    return betas


@nb.njit
def rbf_polynomial(r, n, betas, r_cut):
    """Polynomial radial basis function."""
    poly = 0
    for k in range(1, len(betas) + 1):
        poly += betas[n, k - 1] * (r_cut - np.clip(r, 0, r_cut)) ** (k + 2)
    return poly


@nb.njit
def get_basis_poly(r_cut, n_max, rx):
    fs = np.zeros((n_max, len(rx)))
    betas = get_basis_poly_betas(r_cut, n_max)
    for n in range(1, n_max + 1):
        fs[n - 1, :] = (r_cut - np.clip(rx, 0, r_cut)) ** (n + 2)
    # return fs, betas
    gss = np.dot(betas, fs)
    return fs, gss


betas = get_basis_poly_betas(5, 7)
betas.shape

x = np.arange(100) / 100 * 8
y, gss = get_basis_poly(3, 4, x)
plt.plot(x, gss.T)
plt.xlim(0, 3)
plt.ylim(-1, 3)


from dscribe.descriptors import SOAP

soap = SOAP(
    species=["H"],
    r_cut=3,
    n_max=4,
    l_max=2,
    rbf="polynomial",
)
rx, gss = soap.get_basis_poly(3, 4)
plt.plot(rx, gss.T)

plt.xlim(0, 3)


from scipy.linalg import sqrtm

rx2, gss2 = get_basis_poly(3, 4)
plt.plot(rx2, gss2.T)


def get_basis_poly(rcut, nmax):
    """Used to calculate discrete vectors for the polynomial basis functions.

    Args:
        rcut(float): Radial cutoff.
        nmax(int): Number of polynomial radial bases.

    Returns:
        (np.ndarray, np.ndarray): Tuple containing the evaluation points in
        radial direction as the first item, and the corresponding
        orthonormalized polynomial radial basis set as the second item.
    """
    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    # for more info see: https://doi.org/10.1103/PhysRevB.87.184115 (on representing chemical environments)
    S = np.zeros((nmax, nmax), dtype=np.float64)
    norm_factors = np.zeros(nmax, dtype=np.float64)
    for i in range(1, nmax + 1):
        norm_factors[i - 1] = np.sqrt(rcut ** (2 * i + 5) / (2 * i + 5))
        for j in range(1, nmax + 1):
            S[i - 1, j - 1] = np.sqrt((5 + 2 * j) * (5 + 2 * i)) / (5 + i + j)

    # Get the beta factors that orthonormalize the set with Löwdin
    # orthonormalization
    betas = sqrtm(np.linalg.inv(S)) / norm_factors[None, :]

    # If the result is complex, the calculation is currently halted.    
    if betas.dtype == np.complex128:
        raise ValueError(
            "Could not calculate normalization factors for the radial "
            "basis in the domain of real numbers. Lowering the number of "
            "radial basis functions (nmax) or increasing the radial "
            "cutoff (rcut) is advised."
        )

    # The radial basis is integrated in a very specific nonlinearly spaced
    # grid given by rx
    x = get_rx()

    rCutVeryHard = rcut + 5.0
    rx = rcut * 0.5 * (x + 1)

    # Calculate the value of the orthonormalized polynomial basis at the rx
    # values
    fs = np.zeros([nmax, len(x)])
    for n in range(1, nmax + 1):
        fs[n - 1, :] = (rcut - np.clip(rx, 0, rcut)) ** (n + 2)

    gss = np.dot(betas, fs)

    return rx, gss


rx2, gss2 = get_basis_poly(3, 4)
plt.plot(rx2, gss2.T)