import numpy as np
from quadratures import gaussian_quadrature
import matplotlib.pyplot as plt


def poly3(x):
    return x**3 - 2 * x**2 + 3 * x + 4


def poly3_integral(x):
    return 0.25 * x**4 - 2 / 3 * x**3 + 1.5 * x**2 + 4 * x


def poly20(x):
    return (
        x**20
        - 2 * x**19
        + 3 * x**18
        + 4 * x**17
        + 5 * x**16
        + 6 * x**15
        + 7 * x**14
        + 8 * x**13
        - 9 * x**12
        + 10 * x**11
        + 11 * x**10
        + 12 * x**9
        - 13 * x**8
        + 14 * x**7
        + 15 * x**6
        + 16 * x**5
        - 17 * x**4
        + 18 * x**3
        + 19 * x**2
        - 20 * x
        + 21
    )


def poly20_integral(x):
    return (
        1 / 21 * x**21
        - 2 / 20 * x**20
        + 3 / 19 * x**19
        + 4 / 18 * x**18
        + 5 / 17 * x**17
        + 6 / 16 * x**16
        + 7 / 15 * x**15
        + 8 / 14 * x**14
        - 9 / 13 * x**13
        + 10 / 12 * x**12
        + 11 / 11 * x**11
        + 12 / 10 * x**10
        - 13 / 9 * x**9
        + 14 / 8 * x**8
        + 15 / 7 * x**7
        + 16 / 6 * x**6
        + 17 / 5 * x**5
        + 18 / 4 * x**4
        + 19 / 3 * x**3
        - 20 / 2 * x**2
        + 21 * x
    )


def harmonic_oscillator(x):
    return 0.5 * x**2 * np.sin(x * np.pi * 10)


def harmonic_oscillator_integral(x):
    return (0.0000322515 - 0.0159155 * x**2) * np.cos(10 * np.pi * x) + 0.00101321 * x * np.sin(10 * np.pi * x)





a = 0
b = 20

print(f"Analytical: {poly3_integral(b) - poly3_integral(a)}")
print(f"guassian: {gaussian_quadrature(poly3, a, b, 2)}")

print(f"Analytical: {poly20_integral(b) - poly20_integral(a)}")
print(f"guassian: {gaussian_quadrature(poly20, a, b, 3)}")

print(f"Analytical: {harmonic_oscillator_integral(b) - harmonic_oscillator_integral(a)}")
print(f"guassian: {gaussian_quadrature(harmonic_oscillator, a, b, 25)}")

harmonic_error = np.zeros(100)
for i in range(100):
    analytic = harmonic_oscillator_integral(b) - harmonic_oscillator_integral(a)
    numerical = gaussian_quadrature(harmonic_oscillator, a, b, i + 1)
    harmonic_error[i] = np.abs(analytic - numerical)

plt.plot(np.log10(harmonic_error + 1))


plt.plot(np.linspace(a, b, 100), poly3(np.linspace(a, b, 100)))
plt.plot(np.linspace(a, b, 100), poly20(np.linspace(a, b, 100)), c="r")
plt.plot(np.linspace(a, b, 100), harmonic_oscillator(np.linspace(a, b, 100)), c="g")
