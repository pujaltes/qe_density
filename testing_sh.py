import numpy as np
from quadratures import get_lebedev
from tools_old import project_density, plot_density_projections, get_density_feats
from tools import interpolate_density, cartesian_to_spherical_angles, get_basis_poly, lsq_sph_coeffs, reconstructor, get_basis_gto
import qe_density_reader.reader as qer
from os.path import join as pjoin   

from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from grid.utils import generate_real_spherical_harmonics as grsh



DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
dens = qer.Density(pjoin(DIRPATH, "dsgdb9nsd_130767.hdf5"))
rho, atoms_positions, coords = get_density_feats(dens)  # NOTE: this transposes the density from FORTRAN to C order

r_cut = 5
n_max = 6  # Number of radial basis functions
l_max = 8
cut_pad = 0




def lsq_sph_coeffs(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    centred_coords = coords - atoms_positions[:, None, None, None]
    radii = np.linalg.norm(centred_coords, axis=-1)
    coeffs = np.zeros((len(atoms_positions), n_max * (l_max + 1) ** 2))
    f = []
    masks = []
    for i, ap in enumerate(atoms_positions):
        mask = radii[i] < r_cut
        masks.append(mask)
        c_coords = centred_coords[i][mask]
        r = radii[i][mask]
        c_coords = centred_coords[i][mask]

        rho_vals = rho[mask]
        theta, phi = cartesian_to_spherical_angles(*c_coords.T, r)
        sh_vals = grsh(l_max, theta, phi)
        gto_basis = get_basis_gto(r_cut=r_cut, n_max=n_max, rx=r, l_max=l_max)
        factors = np.zeros((n_max * (l_max + 1) ** 2, gto_basis.shape[-1]))

        for j in range(l_max + 1):
            basis = gto_basis[j]
            idx = (np.arange(2) + j) ** 2
            s_idx = idx * n_max  # scaled idx
            factors[s_idx[0]: s_idx[1]] = np.einsum("ik,jk->ijk", basis, sh_vals[idx[0]: idx[1]]).reshape(-1, gto_basis.shape[-1])



        #poly_basis = get_basis_poly(r_cut=r_cut, n_max=n_max, rx=r, cut_pad=cut_pad)
        #factors = np.einsum("ik,jk->ijk", poly_basis, sh_vals).reshape(n_max * (l_max + 1) ** 2, -1)
        f.append(factors)
        # compute coefficients using least squares
        coeffs[i], residuals, rank, s = lstsq(factors.T, rho_vals)
    return coeffs, f, masks

def lsq_sph_coeffs_mod(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    centred_coords = coords - atoms_positions[:, None, None, None]
    n_atoms = len(atoms_positions)
    c_coords = centred_coords.reshape((n_atoms * centred_coords.shape[1], *centred_coords.shape[2:]))
    c_coords = centred_coords
    radii = np.linalg.norm(c_coords, axis=-1)
    mask = (radii < r_cut).any(axis=0)
    coeffs = np.zeros((len(atoms_positions) * n_max * (l_max + 1) ** 2))
    radii2 = radii[:, mask]
    c_coords2 = c_coords[:, mask]
    rho_vals = rho[mask]


    theta, phi = cartesian_to_spherical_angles(c_coords2[:, :, 0], c_coords2[:, :, 1], c_coords2[:, :, 2], radii2)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)
    sh_vals = grsh(l_max, theta, phi).reshape(-1, (l_max + 1) ** 2)
    poly_basis = get_basis_poly(r_cut=r_cut, n_max=n_max, rx=radii2.reshape(-1), cut_pad=cut_pad).reshape(-1, n_max)
    factors = np.einsum("ij,ik->ijk", poly_basis, sh_vals, optimize=True)
    factors2 = factors.reshape(-1, n_atoms * n_max * (l_max + 1) ** 2)
    # compute coefficients using least squares
    coeffs, residuals, rank, s = lstsq(factors2, rho_vals)
    return coeffs, f, masks


def lsq_sph_coeffs_mod_gto(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    centred_coords = coords - atoms_positions[:, None, None, None]
    n_atoms = len(atoms_positions)
    c_coords = centred_coords.reshape((n_atoms * centred_coords.shape[1], *centred_coords.shape[2:]))
    c_coords = centred_coords
    radii = np.linalg.norm(c_coords, axis=-1)
    mask = (radii < r_cut).any(axis=0)
    coeffs = np.zeros((len(atoms_positions) * n_max * (l_max + 1) ** 2))
    radii2 = radii[:, mask]
    c_coords2 = c_coords[:, mask]
    rho_vals = rho[mask]


    theta, phi = cartesian_to_spherical_angles(c_coords2[:, :, 0], c_coords2[:, :, 1], c_coords2[:, :, 2], radii2)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)
    sh_vals = grsh(l_max, theta, phi).reshape(-1, (l_max + 1) ** 2)
    poly_basis = get_basis_gto(r_cut=r_cut, n_max=n_max, rx=radii2.reshape(-1), l_max=l_max)[0].reshape(-1, n_max)
    factors = np.einsum("ij,ik->ijk", poly_basis, sh_vals, optimize=True)
    factors2 = factors.reshape(-1, n_atoms * n_max * (l_max + 1) ** 2)
    # compute coefficients using least squares
    coeffs, residuals, rank, s = lstsq(factors2, rho_vals)
    return coeffs, f, masks


tst = np.zeros_like(rho)
tst[mask] = f[-1].T @ coeffs[0]
rho2 = rho.copy()
rho2[~mask] = 0 


fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plot_density_projections(project_density(rho2), dens.atoms_positions * 0, dens.alat, ax=ax[0])
plot_density_projections(project_density(tst), dens.atoms_positions * 0, dens.alat, ax=ax[1])
plt.show()

new_rho = reconstructor(f, np.array(masks), coeffs)




fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plot_density_projections(project_density(rho), dens.atoms_positions * 0, dens.alat, ax=ax[0])
plot_density_projections(project_density(new_rho), dens.atoms_positions * 0, dens.alat, ax=ax[1])
plt.show()



def lsq_sph_coeffs(rho, atoms_positions, coords, r_cut, l_max, n_max, cut_pad=0):
    centred_coords = coords - atoms_positions[:, None, None, None]
    radii = np.linalg.norm(centred_coords, axis=-1)
    coeffs = np.zeros((len(atoms_positions), n_max * (l_max + 1) ** 2))
    mask = (radii < r_cut).any(axis=0)


    c_coords = centred_coords[:, mask]
    r = radii[:, mask]
    rho_vals = rho[mask]
    theta, phi = cartesian_to_spherical_angles(*c_coords.T, r)
    sh_vals = grsh(l_max, theta, phi)
    gto_basis = get_basis_gto(r_cut=r_cut, n_max=n_max, rx=r, l_max=l_max)
    factors = np.zeros((n_max * (l_max + 1) ** 2, gto_basis.shape[-1]))

    for j in range(l_max + 1):
        basis = gto_basis[j]
        idx = (np.arange(2) + j) ** 2
        s_idx = idx * n_max  # scaled idx
        factors[s_idx[0]: s_idx[1]] = np.einsum("ik,jk->ijk", basis, sh_vals[idx[0]: idx[1]]).reshape(-1, gto_basis.shape[-1])



    #poly_basis = get_basis_poly(r_cut=r_cut, n_max=n_max, rx=r, cut_pad=cut_pad)
    #factors = np.einsum("ik,jk->ijk", poly_basis, sh_vals).reshape(n_max * (l_max + 1) ** 2, -1)
    f.append(factors)
    # compute coefficients using least squares
        coeffs[i], residuals, rank, s = lstsq(factors.T, rho_vals)
    return coeffs, f, masks

