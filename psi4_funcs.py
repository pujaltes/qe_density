import numpy as np
import qe_density_reader.reader as qer
import psi4
from os.path import join as pjoin


class mol_reader:
    """
    Class to read xyz files and store atomic positions and atom types in seperate lists
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def read_xyz(self, mol_id):
        filename = pjoin(self.dirpath, f"{mol_id}.xyz")
        self.atoms = []
        # self.positions = []
        with open(filename, "r") as f:
            lines = f.readlines()
        self.n_atoms = int(lines[0])
        for line in lines[2:self.n_atoms + 2]:
            atom_sym, x, y, z, charge = line.split()
            self.atoms.append(atom_sym)

    def read_hdf5_coords(self, mol_id):
        filename = pjoin(self.dirpath, f"{mol_id}.hdf5")
        dens = qer.Density(pjoin(self.dirpath, f"{mol_id}.hdf5"))
        self.positions = list(dens.atoms_positions)
        
    def xyz_to_string(self):
        xyz = f"{self.n_atoms}\n\n"
        for atom, pos in zip(self.atoms, self.positions):
            xyz += f"{atom} {pos[0]} {pos[1]} {pos[2]}\n"
        return xyz
    
    def get_psi4_molecule(self, mol_id):
        self.read_xyz(mol_id)
        self.read_hdf5_coords(mol_id)
        xyz_out = self.xyz_to_string()
        self.molecule = psi4.core.Molecule.from_string(xyz_out, dtype="xyz", fix_com=True, fix_orientation=True)
        return self.molecule


class desnity_fitter:
    def __init__(self, mol, rho, coords, basis_set="cc-pVDZ-jkfit"):
        self.mol = mol
        self.rho = rho
        self.coords = coords
        self.basis_set = basis_set

        self._build_primitives()

    def _build_primitives(self):
        self.basis = psi4.core.BasisSet.build(self.mol, "BASIS", self.basis_set)
        self.mints = psi4.core.MintsHelper(self.basis)
        self.S = np.asarray(self.mints.ao_overlap())
        
    


if __name__ == "__main__":
    DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
    mol = mol_reader(DIRPATH)
    mol.get_psi4_molecule("dsgdb9nsd_130767")
    print(mol.molecule)