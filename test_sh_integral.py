import numpy as np
from scipy.special import sph_harm
import numgrid


def gauss_rbf(rx, alpha):
    return np.exp(-alpha * rx**2)


def test_r3_func(r, theta, phi, alpha, coeff):
    return coeff * sph_harm(0, 0, theta, phi) * gauss_rbf(r, alpha)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi



coordinates, ang_weights = numgrid.angular_grid(350)
coordinates = np.array(coordinates)
r, theta, phi = cartesian_to_spherical(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
radii, r_weights = np.polynomial.legendre.leggauss(1000)
radii = (radii + 1) / 2

# from grid.rtransform import BeckeRTransform
from grid.onedgrid import GaussLegendre

# rgrid = BeckeRTransform(0, R=1.5).transform_1d_grid(GaussLegendre(1000))

# rgrid.integrate(gauss_rbf(rgrid.points, alpha))
# radii = rgrid.points
# r_weights = rgrid.weights

sum(r_weights)

alpha = 15.6752927
coeff = 0.01489391

coeffs = [0.01489391     ,  0.04768351  ,     0.08761071 ,      0.063234  ,       0.01352236]
alphas = [15.6752927     ,  3.6063578   ,     1.2080016  ,      0.4726794 ,       0.20181]
norms = [5.614638980502037 ,1.8651411400891058, 0.821222281824621 ,0.4062890324224442, 0.2145937672719459]


integral = 0
for j in range(5):
    coeff = coeffs[j]
    alpha = alphas[j]
    norm = norms[j]
    for i, r in enumerate(radii):
        integral += r_weights[i] * test_r3_func(r, 0, 0, alpha, coeff)
print(integral)


integral = 0
for i, r in enumerate(radii):
    integral += r_weights[i] * np.dot(sph_harm(0, 0, phi, theta), ang_weights)
print(integral)

print(coeff * sph_harm(0, 0, 0, 0) * np.sqrt(np.pi / alpha) / 2)



np.dot(sph_harm(0, 0, phi, theta), ang_weights)

integral = 0
for i, r in enumerate(radii):



coordinates = np.array(coordinates)
r, theta, phi = cartesian_to_spherical(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])


test_r3_func(1, theta, phi, 1, 1)