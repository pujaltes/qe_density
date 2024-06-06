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
        self.molecule.set_units(psi4.core.GeometryUnits.Bohr)
        return self.molecule


class density_fitter:
    def __init__(self, mol, basis_set="cc-pVDZ-jkfit"):
        self.mol = mol
        self.basis_set = basis_set

        self._build_primitives()

    def _build_primitives(self):
        # if not self.verbose:
        psi4.core.be_quiet()
        self.basis = psi4.core.BasisSet.build(self.mol, "BASIS", self.basis_set)
        self.mints = psi4.core.MintsHelper(self.basis)
        self.S = np.asarray(self.mints.ao_overlap())
        # TODO: add number of radial points and angular points
        # psi4.core.get_global_option("DFT_SPHERICAL_POINTS")
        # psi4.core.get_global_option("DFT_RADIAL_POINTS")
        # https://github.com/psi4/psi4/blob/f20a7c61ca0f4939885aa28f96d7d88058a71816/psi4/src/psi4/libfock/cubature.cc#L4271
        self.grid = psi4.core.DFTGrid.build(self.mol, self.basis)
        self.numint = psi4.core.NumIntHelper(self.grid)

    def _get_block_points(self):
        """
        Returns the block points for the grid
        """
        points = np.zeros((self.grid.npoints(), 3))
        blocks = self.grid.blocks()
        idx = 0
        for block in blocks:
            xbl = block.x().to_array()
            ybl = block.y().to_array()
            zbl = block.z().to_array()
            points[idx : idx + xbl.shape[0]] = np.stack([xbl, ybl, zbl], axis=1)
            idx += xbl.shape[0]
        self.points
        return points
            

    def _get_function_values(self, func):
        """
        Returns the values of the function at the grid points
        """
        

    def _compute_overlap_functional(self):
        """
        Computes the overlap functional $\langle F |  f \rangle$=\int f(r) F(r) dr
        for the density f (rho) and the basis function F
        """
        # See: https://forum.psicode.org/t/psi4-core-numinthelper-how-to-use/2773/8
        
    


if __name__ == "__main__":
    DIRPATH = "/home/sp2120/rds/rds-pdb_dist-PDSVOqhVGhM/data/qm9/dsgdb9nsd_atom15_out"
    mol = mol_reader(DIRPATH)
    mol.get_psi4_molecule("dsgdb9nsd_130767")
    zz = density_fitter(mol.get_psi4_molecule("dsgdb9nsd_130767"))
    print(mol.molecule)