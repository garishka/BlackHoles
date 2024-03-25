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

RK45_ACCURACY = 1e-8        # Desired accuracy for the RK45 method
EPSILON = 1e-16     # A small value to prevent division by zero errors
NUM_ITER = int(1e4)      # Maximum number of iterations allowed before termination
#DELTA = 1e-4        # Delta value used for evaluating if the photon trajectory originates from the black hole


@dataclass
class StepErrors:
    # https://doi.org/10.48550/arXiv.1806.0869
    cur_err: float
    prev_err: float
    sec_prev_err: float


err = StepErrors(RK45_ACCURACY, RK45_ACCURACY, RK45_ACCURACY)

#################################### DORMAND-PRINCE, uwu ###############################################################


# func(t, qp, *params)
# init = (q0, p0)
def RK45_mod(func: Callable, init: Union[np.ndarray, List], t_interval: Union[List, tuple], h_init: float, trajectory: bool,
             r_plus: float, delta_r: float, **params) -> tuple:

    if len(params) > (len(inspect.signature(func).parameters)-2):       # -2 за компенсиране на t, y
        warnings.warn("The number of parameters given exceeds the number of positional arguments. "
                      "The excess will be removed.")
        params = {key: value for key, value in params.items() if key in inspect.signature(func).parameters.keys()}
        if len(params) < (len(inspect.signature(func).parameters)-2):
            raise TypeError("Check your parameter names. Something is wrong.")

    # щото ми омръзна от input variables
    # alpha = np.sqrt(1 - (r_plus-1) ** 2)

    rev = bool((t_interval[0] > t_interval[-1]) or (t_interval[0] < 0))

    # не е вярно
    if rev:
        t_interval = sorted(list(t_interval))
    elif t_interval[0] == t_interval[-1]:
        warnings.warn("Check your t intervals.")

    t = np.zeros(shape=NUM_ITER)
    y = np.zeros(shape=(NUM_ITER, 2*len(init)))
    t[0] = t_interval[0]
    y[0] = np.tile(init, 2)
    h = h_init

    k = np.zeros(shape=(7, len(init)), dtype=float)

    j = 1
    while j < NUM_ITER:       # xrange is faster and uses less memory
        for i in range(7):
            k[i] = h * func(t[j-1] + h * c[i], y[j-1, :len(init)] + a[i, :] @ k)

        y[j, :len(init)] = y[j-1, :len(init)] + b4 @ k
        y[j, len(init):] = y[j-1, :len(init)] + b5 @ k

        dif = np.max(np.abs(y[j, :len(init)] - y[j, len(init):]))
        err.sec_prev_err = err.prev_err
        err.prev_err = err.cur_err
        err.cur_err = dif

        # p47, "Numerical Methods" Jeffrey R. Chasnov (lecture notes adapted for Coursera)
        # s = (1e-4 / dif) ** 0.2     # какво е това ϵ във формулата -> desired error tolerance

        if dif < RK45_ACCURACY:
            t[j] = t[j-1] + h
            h *= (RK45_ACCURACY / (err.cur_err + EPSILON)) ** (0.58 / 5)
            h *= (RK45_ACCURACY / (err.prev_err + EPSILON)) ** (-0.21 / 5)
            h *= (RK45_ACCURACY / (err.sec_prev_err + EPSILON)) ** (0.10 / 5)

            if y[j, 1] > init[1] + 1:
                y = y.transpose()[:len(init)]
                y = y[:, :np.count_nonzero(y[1])]
                if trajectory:
                    return False, t, y
                else:
                    return False, y[:, -1]
            elif y[j, 1] < r_plus + delta_r:
                y = y.transpose()[:len(init)]
                y = y[:, :np.count_nonzero(y[1])]
                if trajectory:
                    return True, t, y
                else:
                    return True, y[:, -1]

            j += 1

        else:
            h *= 0.8 * (RK45_ACCURACY / (err.cur_err + 1e-10)) ** 0.25

    y = y.transpose()[:len(init)]
    if trajectory:
        return False, t, y
    else:
        return False, y[:, -1]
