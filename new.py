import numpy as np
from scipy.integrate import tplquad
from scipy.linalg import sqrtm
import scipy
import numba as nb

    



def coefficients_polynomial(system, centers, n_max, r_cut):
    """Used to numerically calculate the inner product coeffientes of SOAP
    with polynomial radial basis.
    """
    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    S = np.zeros((n_max, n_max))
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            S[i - 1, j - 1] = (2 * (r_cut) ** (7 + i + j)) / (
                (5 + i + j) * (6 + i + j) * (7 + i + j)
            )
    betas = sqrtm(np.linalg.inv(S))

    def rbf_polynomial(r, n, l):
        poly = 0
        for k in range(1, n_max + 1):
            poly += betas[n, k - 1] * (r_cut - np.clip(r, 0, r_cut)) ** (k + 2)
        return poly

    return soap_integration(system, centers, args, rbf_polynomial)


def soap_integration(system, centers, n_max, l_max, rbf_function):
    """Used to numerically calculate the inner product coeffientes of SOAP
    with polynomial radial basis.
    """

    positions = system.get_positions()
    atomic_numbers = system.get_atomic_numbers()
    species_ordered = sorted(list(set(atomic_numbers)))
    n_elems = len(species_ordered)

    p_args = []
    p_index = []
    for i, ipos in enumerate(centers):
        for iZ, Z in enumerate(species_ordered):
            indices = np.argwhere(atomic_numbers == Z).flatten()
            elem_pos = positions[indices]
            # This centers the coordinate system at the soap center
            elem_pos -= ipos
            for n in range(n_max):
                for l in range(l_max + 1):
                    for im, m in enumerate(range(-l, l + 1)):
                        p_args.append((args, n, l, m, elem_pos, rbf_function))
                        p_index.append((i, iZ, n, l, im))

    results = Parallel(n_jobs=8, verbose=1)(delayed(integral)(*a) for a in p_args)

    coeffs = np.zeros((len(centers), n_elems, n_max, l_max + 1, 2 * l_max + 1))
    for index, value in zip(p_index, results):
        coeffs[index] = value
    return coeffs


def integral(args, n, l, m, elem_pos, rbf_function):
    r_cut = args["r_cut"]
    sigma = args["sigma"]
    weighting = args.get("weighting")

    # Integration limits for radius
    r1 = 0.0
    r2 = r_cut + 5

    # Integration limits for theta
    t1 = 0
    t2 = np.pi

    # Integration limits for phi
    p1 = 0
    p2 = 2 * np.pi

    def soap_coeff(phi, theta, r):
        # Regular spherical harmonic, notice the abs(m)
        # needed for constructing the real form
        ylm_comp = scipy.special.sph_harm(
            np.abs(m), l, phi, theta
        )  # NOTE: scipy swaps phi and theta

        # Construct real (tesseral) spherical harmonics for
        # easier integration without having to worry about
        # the imaginary part. The real spherical harmonics
        # span the same space, but are just computationally
        # easier.
        ylm_real = np.real(ylm_comp)
        ylm_imag = np.imag(ylm_comp)
        if m < 0:
            ylm = np.sqrt(2) * (-1) ** m * ylm_imag
        elif m == 0:
            ylm = ylm_comp
        else:
            ylm = np.sqrt(2) * (-1) ** m * ylm_real

        # Atomic density
        rho = 0
        ix = elem_pos[:, 0]
        iy = elem_pos[:, 1]
        iz = elem_pos[:, 2]
        ri_squared = ix**2 + iy**2 + iz**2
        rho = np.exp(
            -1
            / (2 * sigma**2)
            * (
                r**2
                + ri_squared
                - 2
                * r
                * (
                    np.sin(theta) * np.cos(phi) * ix
                    + np.sin(theta) * np.sin(phi) * iy
                    + np.cos(theta) * iz
                )
            )
        )
        if weighting:
            weights = get_weights(np.sqrt(ri_squared), weighting)
            rho *= weights
        rho = rho.sum()

        # Jacobian
        jacobian = np.sin(theta) * r**2

        return rbf_function(r, n, l) * ylm * rho * jacobian

    cnlm = tplquad(
        soap_coeff,
        r1,
        r2,
        lambda r: t1,
        lambda r: t2,
        lambda r, theta: p1,
        lambda r, theta: p2,    
        epsabs=1e-6,
        epsrel=1e-4,
    )
    integral, error = cnlm

    return integral