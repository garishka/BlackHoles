from sympy import symbols
import numpy as np
from math_tools.dual import partial_deriv
from typing import Callable


def g00(r, theta, gamma):
    return r * np.sin(theta) ** gamma


def meow_g00(args, params):

    return 1/g00(*args, params)

def meow_g11(args, params):

    return 2/g00(*args, params)


oh = np.frompyfunc(meow_g00, 2, 1)
ho = np.frompyfunc(meow_g11, 2, 1)



def dummy_metric(r, theta, gamma):
    A = lambda r, th, gamma: (1 - 2 / (r * gamma)) ** gamma
    B = lambda r, th, gamma: ((r ** 2 - 2 * r / gamma) / (r ** 2 - 2 * r / gamma + (np.sin(theta) / gamma) ** 2)) ** (gamma ** 2 - 1)
    C = lambda r, th, gamma: (r ** 2 - 2 * r / gamma) ** (gamma ** 2) / (r ** 2 - 2 * r / gamma + (np.sin(theta) / gamma) ** 2) ** (gamma ** 2 - 1)

    g = np.zeros(shape=(4, 4))

    g[0, 0] = lambda r, th, gamma: - A
    g[1, 1] = lambda r, th, gamma: B / A
    g[2, 2] = lambda r, th, gamma: C / A
    g[3, 3] = lambda r, th, gamma: (r ** 2 - 2 * r / gamma) * np.sin(theta) ** 2 / A

    return g
