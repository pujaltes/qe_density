import numpy as np
import scipy.interpolate
from quadratures import get_lebedev, gaussian_quadrature_pts
from tools_old import project_density, plot_density_projections, get_density_feats
from tools import spherical_to_cartesian, get_dens_interpolator
import qe_density_reader.reader as qer
from os.path import join as pjoin   

from scipy.linalg import lstsq
import scipy
import matplotlib.pyplot as plt
from grid.utils import generate_real_spherical_harmonics as grsh


DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
dens = qer.Density(pjoin(DIRPATH, "dsgdb9nsd_130767.hdf5"))

fig, ax = plot_density_projections(project_density(dens.rho), dens.atoms_positions, dens.alat)


r_cut = 5
n_max = 6  # Number of radial basis functions
l_max = 6  # Maximum degree of spherical harmonics

rho, atoms_positions, coords = get_density_feats(dens)  # NOTE: this transposes the density from FORTRAN to C order
# coeffs, f, masks = lsq_sph_coeffs(rho, atoms_positions, coords, r_cut, l_max, n_max)





n_lebedev = 35
n_radial = 10
r_cut = 5

theta, phi, weights = get_lebedev(n_lebedev)
r, r_weights = gaussian_quadrature_pts(0, r_cut, n_radial)

# create spherical grid
r = r[:, None, None] * np.ones((len(theta), len(phi)))
theta = theta[None, :, None] * np.ones([*r.shape])
phi = phi[None, None, :] * np.ones([*r.shape])
r_weights = r_weights[:, None, None] * np.ones([*r.shape])
lebedev_weights = weights[None, :, None] * np.ones([*r.shape])

x, y, z = spherical_to_cartesian(r, theta, phi)

# interp = scipy.interpolate.griddata(coords, rho, (x, y, z), method='linear')






def spherical_integral_grid(lebedev_degree, n_radial, r_cut):
    """
    Generate a spherical grid for integration based on Lebedev quadrature for angular coordinates 
    and Gaussian quadrature for radial coordinates.

    Parameters
    ----------
    lebedev_degree : int
        Degree of the Lebedev quadrature for the angular grid. The actual number of points (n_lebedev) is determined by this degree.
    n_radial : int
        Number of points in the Gaussian quadrature for the radial grid.
    r_cut : float
        Cutoff radius for the radial grid.

    Returns
    -------
    x : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        X-coordinates of the spherical grid points.
    y : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Y-coordinates of the spherical grid points.
    z : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Z-coordinates of the spherical grid points.
    r : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Radial distances of the spherical grid points.
    theta : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Polar angles of the spherical grid points.
    phi : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Azimuthal angles of the spherical grid points.
    r_weights : ndarray of shape (n_radial,)
        Weights for the radial integration.
    lebedev_weights : ndarray of shape (n_radial, n_lebedev, n_lebedev)
        Weights for the angular integration.
    """
    theta, phi, weights = get_lebedev(lebedev_degree)
    r, r_weights = gaussian_quadrature_pts(0, r_cut, n_radial)

    # create spherical grid
    r = r[:, None, None] * np.ones((len(theta), len(phi)))
    theta = theta[None, :, None] * np.ones([*r.shape])
    phi = phi[None, None, :] * np.ones([*r.shape])
    lebedev_weights = weights[None, :, None] * np.ones([*r.shape])

    x, y, z = spherical_to_cartesian(r, theta, phi)
    return x, y, z, r, theta, phi, r_weights, lebedev_weights


x, y, z, r, theta, phi, r_weights, lebedev_weights = spherical_integral_grid(n_lebedev, n_radial, r_cut)
interp = get_dens_interpolator(coords, rho)


def interpolate_around_atom(interpolator, atom_position, x, y, z):
    """
    Interpolate the density around an atom.

    Parameters
    ----------
    interpolator : callable
        Scipy interpolator function that takes a tuple of coordinates as input.
    atom_position : ndarray of shape (3,)
        Position of the atom.
    x : ndarray of shape (n_x, n_y, n_z)
        X-coordinates of the grid points.
    y : ndarray of shape (n_x, n_y, n_z)
        Y-coordinates of the grid points.
    z : ndarray of shape (n_x, n_y, n_z)
        Z-coordinates of the grid points.

    Returns
    -------
    rho_interp : ndarray of shape (n_x, n_y, n_z)
        Interpolated density.
    """
    return interpolator((x + atom_position[0], y + atom_position[1], z + atom_position[2]))

rho_interp = interpolate_around_atom(interp, atoms_positions[0], x, y, z)


def evaluate_spherical_harmonics(l_max, theta, phi):
    """
    Evaluate the real spherical harmonics up to a given degree.

    Parameters
    ----------
    l_max : int
        Maximum degree of the spherical harmonics.
    theta : ndarray of same shape as phi
        Polar angles.
    phi : ndarray of same shape as theta
        Azimuthal angles.

    Returns
    -------
    spherical_harm : list of ndarrays
        List of spherical harmonics evaluated at the input angles.
    """
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape.")
    spherical_harm = grsh(l_max, theta.flatten(), phi.flatten())
    return [x.reshape([*theta.shape]) for x in spherical_harm]




eval_coords = np.array([x.flatten(), y.flatten(), z.flatten()]).T
rho_interp = interp((    
    x + dens.atoms_positions[0, 0],
    y + dens.atoms_positions[0, 1],
    z + dens.atoms_positions[0, 2]
)).reshape([*x.shape])






l_max = 6

spherical_harm = grsh(l_max, theta.flatten(), phi.flatten())
spherical_harm = [x.reshape([*theta.shape]) for x in spherical_harm]




class ElectronDensity():
    def __init__(self, rho, coords, atoms_positions):
        self.rho = rho
        self.coords = coords
        self.atoms_positions = atoms_positions
    
    def setup_integration_grid(self, n_lebedev, n_radial, r_cut):
        self.theta, self.phi, self.weights = get_lebedev(n_lebedev)
        self.r, self.r_weights = gaussian_quadrature_pts(0, r_cut, n_radial)
        # create spherical grid
        r = np.tile(self.r, len(self.theta) * len(self.phi))
        theta = np.repeat(np.tile(self.theta, len(self.phi)), len(self.r))
        phi = np.tile(self.phi, len(self.theta))
        self.x, self.y, self.z = spherical_to_cartesian(self.r

    def 
    