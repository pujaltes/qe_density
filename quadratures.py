import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.special import lpmn, gamma
from grid.angular import AngularGrid
from grid.utils import generate_real_spherical_harmonics
import warnings
from tools import spherical_to_cartesian, cartesian_to_spherical


# fmt: off
LEBEDEV_DIRPATH = "/rds/project/rds-PDSVOqhVGhM/data/quadrature/sphere_lebedev"
AVAILABLE_LEBEDEV = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]  # noqa: E501
# fmt: on


@nb.njit
def legendre_polynomial(x, n):
    """Legendre polynomial of order n."""
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return (2 * n - 1) / n * x * legendre_polynomial(x, n - 1) - (n - 1) / n * legendre_polynomial(x, n - 2)


def gaussian_quadrature(f, a, b, n):
    """Gaussian quadrature for a function f over the interval [a, b] with n nodes."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * np.sum(w * f(0.5 * (b - a) * x + 0.5 * (b + a)))


def get_lebedev(n):
    """Get the Lebedev grid with n nodes."""
    if n not in AVAILABLE_LEBEDEV:
        raise ValueError(f"Lebedev grid with n={n} not available.")
    filepath = f"{LEBEDEV_DIRPATH}/lebedev_{n:03d}.txt"
    theta, phi, weights = np.loadtxt(filepath, unpack=True, dtype=np.float64)
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    return theta, phi, weights


# plot points in 3d using spherical coordinates
def plot_points(x, y, z, weights=None, ax=None):
    """Plot points in 3d using spherical coordinates."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    if weights is None:
        ax.scatter(x, y, z)
    else:
        ax.scatter(x, y, z, s=weights * 50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # change angle of view
    # ax.view_init(30, 90)
    return ax


def gen_sph_orders(lmax):
    """
    Generate the spherical harmonic orders for a given lmax.
    l i degree and m is order
    """
    orders = np.zeros(((lmax + 1) ** 2, 2), dtype=np.int64)
    for l in range(lmax + 1):
        orders[l**2 : (l + 1) ** 2] = l, 0
        for m in range(1, l + 1):
            orders[l**2 + 2 * m - 1] = l, m
            orders[l**2 + 2 * m] = l, -m
    return orders


class LebedevGrid:
    """Class for Lebedev quadrature grids for integration over the unit sphere"""

    def __init__(self, degree=29):
        """Initialize Lebedev grid with n nodes."""
        # TODO: add option to select number of points rather than degree of accuracy
        self.degree, self.n_points = self._verify_degree(degree)
        self.theta, self.phi, self.weights = get_lebedev(self.degree)
        self.n_points = self.theta.shape[0]
        self.points = np.stack(spherical_to_cartesian(1, self.theta, self.phi), axis=0)

    def _verify_degree(self, degree):
        if degree in AVAILABLE_LEBEDEV:
            return degree
        else:
            round_degree = np.array(AVAILABLE_LEBEDEV)[(degree <= np.array(AVAILABLE_LEBEDEV)).argmax()]
            warnings.warn(
                f"Lebedev grid with degree ({degree}) not available. Using next highest degree ({round_degree})."
            )
            return round_degree

    def plot(self, ax=None, use_weights=False):
        """Plot the Lebedev grid."""
        if not use_weights:
            return plot_points(*self.points, ax=ax)
        else:
            return plot_points(*self.points, self.weights, ax=ax)

    def integrate(self, x):
        """
        Integrate a function over the Lebedev grid.

        Parameters
        ----------
        x : array-like
            The values of the function to integrate over the Lebedev grid.
            (i.e. the function evaluated at the grid points)

        Returns
        -------
        float
            The result of the integration over the grid.
        """
        return np.sum(self.weights * x) * 4 * np.pi


def spherical_func(theta, phi):
    """Simple spherical function to test lebedev integration methods."""
    return (np.sin(theta) ** 2) * (np.cos(phi) ** 2)


def spherical_func_integral(theta, phi):
    """
    Analytical integral of the spherical function.
    """
    return np.cos(theta / 10) * np.cos(phi) ** 2


def integrate_lebedev(func, n):
    """Integrate a function over the Lebedev grid with n nodes."""
    theta, phi, weights = get_lebedev(n)
    return np.sum(weights * func(theta, phi)) * 4 * np.pi


def tst_func(theta, phi):
    return 1


def exact_sphere_polynomial_integral(monomial_exponents):
    """
    Exact formula for the integration of multivariate monomial over surface of unit sphere

    How to Integrate A Polynomial Over A Sphere, Gerald B. Folland
    https://doi.org/10.2307/2695802
    """
    monomial_exponents = np.asarray(monomial_exponents)
    if (monomial_exponents % 2 != 0).any():
        # If any of the exponents is odd then the integral is zero due to symmetry
        return 0
    n_vars = len(monomial_exponents)
    betas = (monomial_exponents + 1) / 2
    return 2 * np.prod(gamma(betas)) / gamma(np.sum(betas))


"""
x = generate_real_spherical_harmonics(2, theta, phi)
z = LebedevGrid(29)
z.plot()
z.integrate(x[0])


z = lambda x: x**2


def pol_test(theta, phi):
    x, y, z = spherical_to_cartesian(1, theta, phi)
    return x**2 * y**4 * z**2


theta, phi, weights = get_lebedev(29)
plot_points(phi, theta)

x, y, z = spherical_to_cartesian(1, theta, phi)
plot_points(x, y, z)
(np.linalg.norm([x, y, z], axis=0) == 1).all()

exact_sphere_polynomial_integral([2, 4, 2])
integrate_lebedev(pol_test, 27)  # / (4 * np.pi)


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x, y, z)
ax.set_title(f"Angular Grid Points of Degree {29}")
plt.show()


ang_grid = AngularGrid(degree=29)
x1, y2, z3 = ang_grid.points.T
plot_points(x1, y2, z3)


integrate_lebedev


integrate_lebedev(tst_func, 27)  # / (4 * np.pi)
integrate_lebedev(spherical_func, 29)
spherical_func_integral(2 * np.pi, np.pi) - spherical_func_integral(0, 0)

# plotting the spherical harmonics
# https://scipython.com/book/chapter-8-scipy/examples/visualizing-the-spherical-harmonics/
# http://keatonb.github.io/archivers/shanimate
# https://math.stackexchange.com/questions/145080/how-are-the-real-spherical-harmonics-derived
# https://people.sc.fsu.edu/~lb13f/projects/space_environment/egm96.php
# https://mattermodeling.stackexchange.com/questions/10901/becke-partitioning-computing-integrals-with-lebedev-and-gauss-chebyshev-quadra


# Exact formula for the integration of a polynomial over the surface of a sphere
# See G.Folland, The American Mathematical Monthly, Vol. 108, No. 5 (May, 2001), pp. 446-448.

theta, phi, weights = get_lebedev(113)
plot_points(theta, phi, weights=weights * weights.shape[0])

plot_points(theta, phi, weights=spherical_func(theta, phi))
plt.scatter(theta, phi, s=spherical_func(theta, phi) * 10)
weights.sum()
"""
