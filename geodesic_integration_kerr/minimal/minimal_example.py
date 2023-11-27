import itertools
import logging
import numpy as np
from geodesic_integration_kerr import dual
from PIL import Image, ImageDraw
from geodesic_integration_kerr.integrator import symplectic_integrator
import time
import concurrent.futures

res = 80
r0 = 500
th0 = np.pi/2
a0 = 0.99
alpha_values = np.linspace(-5e-3*np.pi, 5e-3*np.pi, res)
beta_values = np.linspace((1-5e-3)*np.pi, (1+5e-3)*np.pi, res)

r_plus = 1 + np.sqrt(1 - a0 ** 2)


# metric
def sigma_expr(r, theta, a):
    return r ** 2 + (a * np.cos(theta)) ** 2


def delta_expr(r, a):
    return r ** 2 - 2 * r + a ** 2

def A_expr(r, theta, a):
    return (r ** 2 + a ** 2) ** 2 - delta_expr(r, a) * (a * np.sin(theta)) ** 2


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

# inverse metric elements
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


def hamiltons_eqs(l, qp, *params):
    t, r, th, phi, p0, p1, p2, p3 = qp
    a = params[0]

    dq0dl = g00(r, th, a) * p0 + g03(r, th, a) * p3    #dual.partial_deriv(hamiltonian, qp, 0, a)
    dq1dl = g11(r, th, a) * p1    #dual.partial_deriv(hamiltonian, qp, 1, a)
    dq2dl = g22(r, th, a) * p2    #dual.partial_deriv(hamiltonian, qp, 2, a)
    dq3dl = g03(r, th, a) * p0 + g33(r, th, a) * p3    #dual.partial_deriv(hamiltonian, qp, 3, a)

    dp0dl = 0 # - dual.partial_deriv(hamiltonian, qp, 4, a)
    dp1dl = - dual.partial_deriv(hamiltonian, qp, 5, a)
    dp2dl = - dual.partial_deriv(hamiltonian, qp, 6, a)
    dp3dl = 0 # - dual.partial_deriv(hamiltonian, qp, 7, a)

    return [dq0dl, dq1dl, dq2dl, dq3dl, dp0dl, dp1dl, dp2dl, dp3dl]


def jacobian(l, qp, *params):
    a = params[0]

    j = np.zeros(shape=(8, 8))

    # да поправя реда на диференциране
    for i in range(8):
        for k in range(4):
            j[k, i] = dual.second_partial_deriv(hamiltonian, qp, [k, i], a)
        for k in range(4, 8):
            j[k, i] = - dual.second_partial_deriv(hamiltonian, qp, [k, i], a)

    return j


def init_p(r, th, a, alpha_i, beta_i):
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

    results = symplectic_integrator(hamiltonian, qp, [a0], 0.1, 1, 10_000)

    for k in range(len(results[0])):
        # Δr за данните?
        if results[1, k] <= r_plus + 1e-1:
            # The light ray falls into the black hole; return the values of the pixel that has to be set black
            # С concurrent.futures.ProcessPoolExecutor() не знам как да акумулира промените вместо да създава нова
            # картинка на всеки 7-8 процеса
            return True, (i, j)
    return False, (i, j)


if __name__ == "__main__":
    image = Image.new("RGB", (res, res), "white")
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    logging.basicConfig(level=logging.INFO)
    iterable = list(itertools.product(range(res), range(res)))
    i, j = zip(*iterable)

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(solve_ivp, i, j):
            fallen, coords = result

            if fallen:
                a, b = coords
                pixels[a, b] = (0, 0, 0)

    end = time.time()
    print(f"Time taken: {end - start}")

    image.save("miniBHtest.jpg")
