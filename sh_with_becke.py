"""
We will use atoms in molecules theory partitioning to first seperate the electron denstiies into
atomic densities. Essentially we weight the electron density based on the relative atomic volumes
of the atoms in the molecule and then integrate over that adjusted density to get the spherical
harmonic coefficient expansion of the density.



compute Becke partition using bragg



How to continue DFT calculations from SH decomposition of density:
https://github.com/psi4/psi4/issues/3070

Spherical Harmonic Decomposition of the Density:
https://github.com/theochem/grid/issues/11
"""

import numpy as np
from tools import get_dens_interpolator
from tools_old import get_density_feats
from grid.utils import get_cov_radii
from grid.becke import BeckeWeights
import qe_density_reader.reader as qer
from os.path import join as pjoin
from iodata import load_one


DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
dens = qer.Density(pjoin(DIRPATH, "dsgdb9nsd_130767.hdf5"))
rho, atoms_positions, coords = get_density_feats(dens)  # NOTE: this transposes the density from FORTRAN to C order
atom_type = dens.type_idx
mol = load_one(pjoin(DIRPATH, "dsgdb9nsd_130767.xyz"))
atom_nums = mol.atnums


def covalent_radius_dict(rad_type="alvarez"):
    """
    Returns a dictionary of covalent radii for each element.
    """
    cov_rad = {
        "alvarez": get_cov_radii(np.arange(1, 97), "alvarez"),  
        "bragg": get_cov_radii(np.arange(1, 87), "bragg"),
        "cambridge": get_cov_radii(np.arange(1, 87), "cambridge"),
        }
    return dict([(i + 1, radius) for i, radius in enumerate(cov_rad[rad_type])])

radii = covalent_radius_dict("alvarez")



becke_w = BeckeWeights(radii=radii, order=3)


data = get_cov_radii(np.arange(1, 87, 1), "bragg")
_radii = dict([(i + 1, radius) for i, radius in enumerate(data)])
