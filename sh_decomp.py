import numpy as np
import numba as nb


@nb.njit
def compute_distances(coords, atoms_positions):
    """
    Compute the distances between each grid point and each atom.

    Parameters:
    - coords: 4D array of grid point coordinates.
    - atoms_positions: 2D array of atom positions.

    Returns:
    - 3D array of distances.
    make this vectorized
    """
    distances = np.zeros((coords.shape[1], coords.shape[2], coords.shape[3], atoms_positions.shape[0]))

    for i in range(coords.shape[1]):
        for j in range(coords.shape[2]):
            for k in range(coords.shape[3]):
                for l in range(atoms_positions.shape[0]):
                    distances[i, j, k, l] = np.linalg.norm(coords[:, i, j, k] - atoms_positions[l, :])

    return distances

@nb.njit
def integrate_samples(x_values, y_values, z_values, f_values):
    """
    Numerically integrate a function represented by samples using the trapezoidal rule.
    Assumes evenly spaced points in each dimension.

    Parameters:
    - x_values, y_values, z_values: 1D arrays of evenly spaced points in x, y, z dimensions.
    - f_values: 3D array of function values at the corresponding (x, y, z) points.

    Returns:
    - Numerical integral result.
    """
    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]
    dz = z_values[1] - z_values[0]

    integral_result = 0.0

    for i in range(len(x_values) - 1):
        for j in range(len(y_values) - 1):
            for k in range(len(z_values) - 1):
                # Trapezoidal rule for each small cube in the grid
                f_avg = 0.125 * (
                    f_values[i, j, k]
                    + f_values[i + 1, j, k]
                    + f_values[i, j + 1, k]
                    + f_values[i + 1, j + 1, k]
                    + f_values[i, j, k + 1]
                    + f_values[i + 1, j, k + 1]
                    + f_values[i, j + 1, k + 1]
                    + f_values[i + 1, j + 1, k + 1]
                )
                integral_result += f_avg * dx * dy * dz

    return integral_result
