"""
Ray Tracing Simulation for a Kerr Black Hole
--------------------------------------------

This script simulates the trajectory of light rays around a Kerr Black Hole and produces an image of the observed
celestial sphere.

Parameters:
-----------
- res (int): Image resolution.
- r0 (float): Initial radial coordinate.
- th0 (float): Initial polar coordinate.
- a0 (float): Spin parameter of the black hole.
- alpha_values (numpy.ndarray): Vertical coordinates of the camera's view.
- beta_values (numpy.ndarray): Horizontal coordinates of the camera's view.
- step (float): Step size for the symplectic integrator.
- omg (float): Coupling constant for the doubled phase space used in the integration scheme.
- steps (int): Number of integration steps.
- r_plus (float): Black Hole event horizon.

Functions:
----------
- sigma_expr(r, theta, a): Expression for sigma, a metric component.
- delta_expr(r, a): Expression for delta, a metric component.
- A_expr(r, theta, a): Expression for A, a metric component.
- g00(r, th, a), g03(r, th, a), g30(r, th, a), g11(r, th, a), g22(r, th, a), g33(r, th, a): Nonzero covariant metric components.
- inv_g00(r, th, a), inv_g03(r, th, a), inv_g30(r, th, a), inv_g11(r, th, a), inv_g22(r, th, a), inv_g33(r, th, a):
Nonzero inverse metric components.
- hamiltonian(qp, a): Hamiltonian for m=0.
- init_p(r, th, a, alpha_i, beta_i): Initialize 4-momentum at (-β_i, α_i).
- solve_ivp(i, j): Solve the initial value problem for the given pixel coordinates (i, j).

Main Execution:
---------------
- The script initializes an image with a white background.
- It uses ray tracing and symplectic integration to determine the trajectory of light rays.
(- Dual numbers are used in the symplectic integrator scheme to increase accuracy.)
- Concurrent processing is employed to enhance computational efficiency.

Note: Adjust the parameters and functions as needed for specific scenarios.
"""


import itertools
import logging
import numpy as np
from PIL import Image, ImageDraw
import time
import concurrent.futures

from geodesic_integration_kerr.integrator import symplectic_integrator

res = 100
r0 = 1_000
th0 = np.pi/2
a0 = 0.99
alpha_values = np.linspace(-3e-3*np.pi, 3e-3*np.pi, res)
beta_values = np.linspace((1-3e-3)*np.pi, (1+3e-3)*np.pi, res)

step = 0.2
omg = 1
steps = 10_000

r_plus = 1 + np.sqrt(1 - a0 ** 2)


def sigma_expr(r, theta, a):
    return r ** 2 + (a * np.cos(theta)) ** 2


def delta_expr(r, a):
    return r ** 2 - 2 * r + a ** 2

def A_expr(r, theta, a):
    return (r ** 2 + a ** 2) ** 2 - delta_expr(r, a) * (a * np.sin(theta)) ** 2


# nonzero covariant metric components
def g00(r, th, a):
    return 2 * r / sigma_expr(r, th, a) - 1
def g03(r, th, a):
    return -2 * r * a * np.sin(th) ** 2 / sigma_expr(r, th, a)
def g30(r, th, a):
    return g03(r, th, a)
def g11(r, th, a):
    return sigma_expr(r, th, a) / delta_expr(r, a)
def g22(r, th, a):
    return sigma_expr(r, th, a)
def g33(r, th, a):
    return A_expr(r, th, a) * np.sin(th) ** 2 / sigma_expr(r, th, a)


# nonzero inverse metric components
def inv_g00(r, th, a):
    return - A_expr(r, th, a) / (delta_expr(r, a) * sigma_expr(r, th, a))
def inv_g03(r, th, a):
    return - 2 * r * a / (delta_expr(r, a) * sigma_expr(r, th, a))
def inv_g30(r, th, a):
    return inv_g03(r, th, a)
def inv_g11(r, th, a):
    return delta_expr(r, a) / sigma_expr(r, th, a)
def inv_g22(r, th, a):
    return 1/sigma_expr(r, th, a)
def inv_g33(r, th, a):
    return (delta_expr(r, a) - (a * np.sin(th)) ** 2) / (delta_expr(r, a) * sigma_expr(r, th, a))


def hamiltonian(qp, a):
    r, th = qp[1:3]
    p0, p1, p2, p3 = qp[4:]
    return 0.5 * (inv_g00(r, th, a) * p0 ** 2 + 2 * inv_g03(r, th, a) * p0 * p3 + inv_g11(r, th, a) * p1 ** 2 + \
        inv_g22(r, th, a) * p2 ** 2 + inv_g33(r, th, a) * p3 ** 2)


def init_p(r: float, th: float, a: float, alpha_i: float, beta_i: float):
    gamma_g = -g03(r, th, a) / np.sqrt(g33(r, th, a) * (g03(r, th, a) ** 2 - g00(r, th, a) * g33(r, th, a)))
    zeta = np.sqrt(g33(r, th, a) / (g03(r, th, a) ** 2 - g00(r,th, a) * g33(r, th, a)))

    p_th = np.sqrt(g22(r, th, a)) * np.sin(alpha_i)
    p_r = np.sqrt(g11(r, th, a)) * np.cos(beta_i) * np.cos(alpha_i)
    L = np.sqrt(g33(r, th, a)) * np.sin(beta_i) * np.cos(alpha_i)
    E = (1 + gamma_g * np.sqrt(g33(r, th, a)) * np.sin(beta_i) * np.cos(alpha_i)) / zeta

    return [E, p_r, p_th, L]


def solve_ivp(i, j):
    logging.info(f"Processing pixel ({i}, {j})")

    p0 = init_p(r0, th0, a0, alpha_values[i], beta_values[j])
    qp = np.asarray(([0, r0, th0, 0] + p0))

    results = symplectic_integrator(hamiltonian, qp, [a0], step, omg, steps)

    for k in range(len(results[0])):
        # Δr за данните?
        if results[1, k] <= r_plus + 1e-1:
            # The light ray falls into the black hole; return the values of the pixel that has to be set black
            # С concurrent.futures.ProcessPoolExecutor() не знам как да акумулира промените вместо да създава нова
            # картинка на всеки 7-8 процеса -> създавам картинката в __main__ частта
            return True, (i, j)
    return False, (i, j)


# TODO: да добавя map-ване към картинка с шарка за небесната сфера
# TODO: да добавя графики на траекторията на няколко лъча и графики на измененинето на запазващите се величини
if __name__ == "__main__":
    image = Image.new("RGB", (res, res), "white")
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    logging.basicConfig(level=logging.INFO)
    iterable = list(itertools.product(range(res), range(res)))
    i, j = zip(*iterable)

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(solve_ivp, i, j):
            fallen, coords = result

            a, b = coords
            if fallen:
                pixels[b, res-1-a] = (0, 0, 0)
            else:
                pixels[b, res-1-a] = (255, 255, 255)

    end = time.time()
    elapsed_time_sec = end - start
    hours, remainder = divmod(elapsed_time_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {hours}:{minutes}:{seconds}")

    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image.save("miniBHtest.png")
