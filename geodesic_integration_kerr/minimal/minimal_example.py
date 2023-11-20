import numpy as np
from geodesic_integration_kerr import dual
from scipy.integrate import solve_ivp
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

res = 50
r0 = 5000
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
    a = params[0]

    dq0dl = dual.partial_deriv(hamiltonian, qp, a, 0)
    dq1dl = dual.partial_deriv(hamiltonian, qp, a, 1)
    dq2dl = dual.partial_deriv(hamiltonian, qp, a, 2)
    dq3dl = dual.partial_deriv(hamiltonian, qp, a, 3)

    dp0dl = - dual.partial_deriv(hamiltonian, qp, a, 4)
    dp1dl = - dual.partial_deriv(hamiltonian, qp, a, 5)
    dp2dl = - dual.partial_deriv(hamiltonian, qp, a, 6)
    dp3dl = - dual.partial_deriv(hamiltonian, qp, a, 7)

    return [dq0dl, dq1dl, dq2dl, dq3dl, dp0dl, dp1dl, dp2dl, dp3dl]


def jacobian(l, qp, *params):
    a = params[0]
    H = hamiltonian(qp, a)

    j = np.zeros(shape=(8, 8))

    for i in range(8):
        for k in range(8):
            j[i, k] = dual.partial_deriv()

    return j


def init_p(r, th, a, alpha_i, beta_i):
    gamma_g = -g03(r, th, a) / np.sqrt(g33(r, th, a) * (g03(r, th, a) ** 2 - g00(r, th, a) * g33(r, th, a)))
    zeta = np.sqrt(g33(r, th, a) / (g03(r, th, a) ** 2 - g00(r,th, a) * g33(r, th, a)))

    p_th = np.sqrt(g22(r, th, a)) * np.sin(alpha_i)
    p_r = np.sqrt(g11(r, th, a)) * np.cos(beta_i) * np.cos(alpha_i)
    L = np.sqrt(g33(r, th, a)) * np.sin(beta_i) * np.cos(alpha_i)
    E = (1 + gamma_g * np.sqrt(g33(r, th, a)) * np.sin(beta_i) * np.cos(alpha_i)) / zeta

    return [E, p_r, p_th, L]


image = Image.new("RGB", (res, res), "white")
draw = ImageDraw.Draw(image)
pixels = image.load()

for i in range(len(alpha_values)):
    for j in range(len(beta_values)):

        print(i, j)

        p0 = init_p(r0, th0, a0, alpha_values[i], beta_values[j])
        qp = [0, r0, th0, 0] + p0
        qp = np.asarray(qp)

        #ne raboti ;((((((((((((((((((((((
        print(hamiltons_eqs(0, qp, a0))
        sol = solve_ivp(hamiltons_eqs,
                        t_span=[0, -np.infty],
                        y0=qp,
                        method="BDF",
                        t_eval=np.linspace(0, -50_000, 100_000),
                        args=(a0,)
                        )


image.save("minimal_BHtest.png")
