import numpy as np
from math_tools import dual


def f0(r, theta):
    pass


def f1(r, theta):
    pass


def f2(r, theta):
    pass


def w(r, theta):
    pass


def christoffel(r, theta):
    W = w(r, theta)
    F0 = f0(r, theta)
    F1 = f1(r, theta)
    F2 = f2(r, theta)
    dWdr = dual.partial_deriv(w, [r, theta], 0)

    gamma = np.zeros(shape=(4, 4, 4), dtype=float)

    a_expr = W * r ** 4 * np.sin(theta) ** 2 * np.exp(2 * F2) * (W / r + 0)