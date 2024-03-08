# M = G = c = 1
import numpy as np


def A_expr(r, gamma):
    return (1 - 2 / (r * gamma)) ** gamma


def B_expr(r, theta, gamma):
    return ((r ** 2 - 2 * r / gamma) / (r ** 2 - 2 * r / gamma + (np.sin(theta) / gamma) ** 2)) ** (gamma ** 2 - 1)


def C_expr(r, theta, gamma):
    return (r ** 2 - 2 * r / gamma) ** (gamma ** 2) / (r ** 2 - 2 * r / gamma + (np.sin(theta) / gamma) ** 2) ** (gamma ** 2 - 1)


def g_cov(r, theta, gamma):
    A = A_expr(r, gamma)
    B = B_expr(r, theta, gamma)
    C = C_expr(r, theta, gamma)

    g = np.zeros(shape=(4, 4), dtype=float)

    g[0, 0] = - A
    g[1, 1] = B / A
    g[2, 2] = C / A
    g[3, 3] = (r ** 2 - 2 * r / gamma) * np.sin(theta) ** 2 / A

    return g

