import numpy as np
from typing import Callable, Union, List
import inspect
import warnings
from dataclasses import dataclass

########################################## BUTCHER TABLEAU #############################################################

a = np.asarray([[0, 0, 0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]], dtype=float)

c = np.asarray([0, 0.2, 0.3, 0.8, 8 / 9, 1, 1], dtype=float)

b4 = np.asarray([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=float)
b5 = np.asarray([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], dtype=float)

################################### ACCURACY AND ERRORS ################################################################

KDP_ACCURACY = 1e-4
EPSILON = 1e-16


@dataclass
class StepErrors:
    cur_err: float
    prev_err: float
    sec_prev_err: float


err = StepErrors(KDP_ACCURACY, KDP_ACCURACY, KDP_ACCURACY)  # това изглежда мега тъпо като решение


#################################### DORMAND-PRINCE, uwu ###############################################################

# func(t, qp, *params)
# init = (q0, p0)
def kdp45(func: Callable, init: Union[np.ndarray, List], t_init: float, h_init: float, num_iter: int, **params) -> tuple:

    if len(params) > (len(inspect.signature(func).parameters)-2):       # -2 за компенсиране на t, y
        warnings.warn("The number of parameters given exceeds the number of positional arguments. "
                      "The excess will be removed.")
        params = {key: value for key, value in params.items() if key in inspect.signature(func).parameters.keys()}
        if len(params) < (len(inspect.signature(func).parameters)-2):
            raise TypeError("Check your parameter names. Something is wrong.")

    t = np.zeros(shape=num_iter)
    y = np.zeros(shape=(num_iter, 2*len(init)))
    t[0] = t_init
    y[0] = np.tile(init, 2)
    h = h_init

    k = np.zeros(shape=(7, len(init)), dtype=float)

    for j in range(1, num_iter):
        for i in range(7):
            k[i] = h * func(t[j-1] + h * c[i], y[j-1, :len(init)] + np.dot(a[i], k), *params.values())

        y[j, :len(init)] = y[j-1, :len(init)] + b4 @ k
        y[j, len(init):] = y[j-1, :len(init)] + b5 @ k

        dif = np.max(np.abs(y[j, :len(init)] - y[j, len(init):]))
        err.sec_prev_err = err.prev_err
        err.prev_err = err.cur_err
        err.cur_err = dif

        # p47, "Numerical Methods" Jeffrey R. Chasnov (lecture notes adapted for Coursera)
        # s = (1e-4 / dif) ** 0.2     # какво е това ϵ във формулата -> desired error tolerance

        if dif < KDP_ACCURACY:
            t[j] = t[j-1] + h
            h *= (KDP_ACCURACY / (err.cur_err + EPSILON)) ** (0.58 / 5)
            h *= (KDP_ACCURACY / (err.prev_err + EPSILON)) ** (-0.21 / 5)
            h *= (KDP_ACCURACY / (err.sec_prev_err + EPSILON)) ** (0.10 / 5)
        else:
            t[j] = t[j-1]
            y[j, :] = y[j-1, :]
            h *= 0.8 * (KDP_ACCURACY / (err.cur_err + 1e-10)) ** 0.25

    y = y.transpose()
    return t, y
