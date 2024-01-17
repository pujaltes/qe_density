import qe_density_reader.reader as qer
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os


def project_density(rho, proj_method="max"):
    """
    Project the density on every plane of the cell.
    """
    proj_func = {"mean": lambda x, axis: np.mean(x, axis=axis), "max": lambda x, axis: np.max(x, axis=axis)}
    projected_density = np.zeros((3,) + rho.shape[:2])
    for i in range(3):
        # numpy uses zyx order so we have to transpose the array before the projection
        # https://stackoverflow.com/questions/46855793/understanding-axes-in-numpy
        projected_density[i] = proj_func[proj_method](rho.T, -1 - i)
    return projected_density


def plot_density_projections(projected_density, atom_positions, alat, ax=None, **kwargs):
    """
    Plot the projections of the density on every plane of the cell.
    """
    axis_labels = np.array(["x", "y", "z"])
    vmax = np.max(projected_density)
    vmin = np.min(projected_density)
    slices = [[0, 1], [0, 2], [1, 2]]
    extent = np.array([0, 1, 0, 1]) * alat
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = ax.flatten()[0].get_figure()
    for i in range(3):
        # Transpose to get the orientation that we want
        ax[i].imshow(projected_density[i].T, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        if atom_positions is not None:
            ax[i].scatter(
                atom_positions[:, slices[i][0]], atom_positions[:, slices[i][1]], color="red", marker="x"
            )
        ax[i].set_xlabel(axis_labels[slices[i][0]])
        ax[i].set_ylabel(axis_labels[slices[i][1]])
        # ax[i].set_title(f"Projection on the {''.join(axis_labels[slices[i]])} plane")
    return fig, ax


def get_coords(rho_shape, cell):
    """
    Get the coordinates of the grid points.
    """
    grid = np.array(
        np.meshgrid(
            np.arange(0, rho_shape[0]),
            np.arange(0, rho_shape[1]),
            np.arange(0, rho_shape[2]),
            indexing="ij",
        )
    )
    coords = np.tensordot(grid / np.array(rho_shape)[:, None, None, None], cell, axes=(0, 1))
    return coords


def get_density_feats(dens):
    """Helper function to get the density features for easier manipulation."""
    coords = get_coords(dens.rho.shape, dens.cell)
    return dens.rho.T.copy(), dens.atoms_positions, coords


# DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
# dens = qer.Density(pjoin(DIRPATH, "dsgdb9nsd_130767.hdf5"))
# dens.rho
# dens.rot
# dens.trans
# dens.atoms_positions
# dens.cell
# trans_mat = dens.rot @ dens.trans
# dens.trans @ dens.rot
# np.array([1, 1, 1, 1]) @ trans_mat
# xmax = ymax = zmax = 2
# xmin = ymin = zmin = -2
# matrix = np.diagflat(np.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), np.float32, order="C"))
# matrix[0:3, 3] = ((xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0)
# dens.cell

# plot_density_projections(project_density(dens.rho), dens.atoms_positions, dens.alat)
