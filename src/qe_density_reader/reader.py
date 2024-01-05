import h5py
import numpy as np
import math
import k3d
import k3d.transform


class Density:
    def __init__(self, fname: str, verbose: bool = False):
        """
        returns a numpy array given the hdf5 file written by the routine

        https://gitlab.com/rikigigi/q-e/-/blob/write_rhor_hdf5/Modules/io_base.f90?ref_type=heads#L995
        """
        self.hdf5file = h5py.File(fname, "r")
        self.rho_data = self.hdf5file.get("rho_of_r_data")
        self.alat = self.rho_data.attrs["alat"]
        self.cell = self.rho_data.attrs["cell_vectors"] * self.alat
        self.atoms_positions = self.rho_data.attrs["tau"] * self.alat
        self.rho_shape = self.rho_data.attrs["grid_dimensions"]

        atoms_types = self.rho_data.attrs["ityp"]
        self.compute_transformation()
        self.rho = self.unpack_rho()
        if verbose:
            print(self.data_info(fname))

        self.V = np.linalg.det(self.cell)
        self.nr = np.array(self.rho.shape).prod()
        self.type_idx = self.rho_data.attrs["ityp"] - 1
        self.atomic_charges = self.rho_data.attrs["zv"][self.type_idx]
        self.dipole_total = None

    def unpack_rho(self):
        rho_index_data_np = np.array(self.hdf5file.get("rho_of_r_index_size"))
        rho_data_np = np.array(self.rho_data)

        rho = np.zeros(self.rho_shape[::1])
        count = 0
        # NOTE: in qe I have the option to not save density values lower than a threshold to save some disk space.
        for idx, size in zip(rho_index_data_np[0::2], rho_index_data_np[1::2]):
            # NB: the index from the file are fortran one, so the first element is 1.
            # But in python indexes starts from 0, so we have the "-1"
            rho.flat[idx - 1 : idx - 1 + size] = rho_data_np[count : count + size]
            count += size

        return rho

    def compute_transformation(self):
        self.rot = np.identity(4)
        self.trans = np.identity(4)
        self.rot[:3, :3] = self.cell.transpose()
        self.trans[:3, 3] = np.ones(3) / 2

    def data_info(self, fname):
        rho_index_data_np = np.array(self.hdf5file.get("rho_of_r_index_size"))
        rho_data_np = np.array(self.rho_data)
        info = f"""
            fname={fname}:
            items: {list(self.hdf5file.items())}
            attrs: {list(self.rho_data.attrs.items())}
            compress_ratio: {(rho_data_np.size + rho_index_data_np.size) / self.rho.size}
            \n rho_index_data_np: {rho_index_data_np}
        """
        return info

    def idx2r(self, *args):
        ijk = np.array(args) / self.rho_shape
        # cell vectors are cell[0],cell[1],cell[2]
        return ijk @ self.cell

    def get_coords(self):
        """
        Get the coordinates of the grid points.
        """
        grid = np.array(
            np.meshgrid(
                np.arange(0, self.rho_shape[0]),
                np.arange(0, self.rho_shape[1]),
                np.arange(0, self.rho_shape[2]),
                indexing="ij",
            )
        )
        coords = np.tensordot(grid / self.rho_shape[:, None, None, None], self.cell, axes=(0, 1))
        return coords

    def dipole(self):
        """
        dipole = \sum_{ijk} r_{ijk} \rho_{ijk} dV
        warning: the molecule must be in the center of the cell,
        and the charge must be zero near the boundaries or the calculation is ill-defined
        """
        if np.any(self.cell != self.cell.transpose()):
            raise NotImplementedError("Only orthogonal cells are implemented")
        ang = 0.52917721  # bohr radius
        e_ang2D = 0.2081943  # https://en.wikipedia.org/wiki/Debye
        r0 = 0.0
        # atomic dipole
        # atomic dipole in electrons charges times Angstrom
        atomic_dipole = np.tensordot(self.atoms_positions * ang - r0, self.atomic_charges, axes=((0,), (0,)))
        print("dipole nuclei, D", atomic_dipole * e_ang2D)

        # electronic dipole is equivalent to the following:
        # dip=0
        # dV=np.linalg.det(cell)/nr
        # for i in range(rho_full.shape[0]):
        #    for j in range(rho_full.shape[1]):
        #        for k in range(rho_full.shape[2]):
        #            r = idx2r(i,j,k)
        #            dip+=r*rho_full[i,j,k]*dV

        dV = self.V / self.nr
        # coord grid
        coords = self.get_coords() * r0 * ang
        
        el_dipole = np.sum((coords) * self.rho[..., None], axis=(0, 1, 2)) * dV
        # dipole
        print("dipole electrons, D", el_dipole * e_ang2D, dV, self.V, self.nr)
        dipole_total = (-el_dipole + atomic_dipole) * e_ang2D
        print(dipole_total)
        self.dipole_total = dipole_total
        return np.linalg.norm(dipole_total)

    def display(self, alpha_coef=15):
        transform = k3d.transform(custom_matrix=self.rot @ self.trans)
        rho_full_draw = k3d.volume(self.rho, alpha_coef=alpha_coef)
        transform.add_drawable(rho_full_draw)
        transform.parent_updated()
        rho_full_draw.transform = transform
        p = k3d.plot()
        p += k3d.points(self.atoms_positions, point_size=0.5)
        p += rho_full_draw
        return p.display()

    def write_compressed_hdf5(self, output_file: str):
        # Open the input HDF5 file in read
        with h5py.File(output_file, "w") as f_out:
            # Iterate over all groups and datasets in the input file
            def copy_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Create a new dataset in the output file with the same name, shape, and datatype
                    dset_out = f_out.create_dataset(
                        name,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        compression="gzip",
                        compression_opts=9,
                        shuffle=True,
                    )
                    # Copy the data from the input dataset to the output dataset with compression
                    dset_out[...] = obj[...]
                    for attr_name, attr_value in obj.attrs.items():
                        dset_out.attrs[attr_name] = attr_value

            self.hdf5file.visititems(copy_dataset)
            if self.dipole_total is not None:
                try:
                    dset = f_out.create_dataset("dipole_total", data=self.dipole_total)
                except Exception as e:
                    print(e)
