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

RK45_ACCURACY = 1e-5
EPSILON = 1e-16
NUM_ITER = 10 ** 3
DELTA = 1e-4


@dataclass
class StepErrors:
    cur_err: float
    prev_err: float
    sec_prev_err: float


err = StepErrors(RK45_ACCURACY, RK45_ACCURACY, RK45_ACCURACY)  # това изглежда мега тъпо като решение

############################################# BH shit ##################################################################


def b_values(alpha):
    r1 = 2 * (1 + np.cos(2 * np.arccos(-alpha) / 3))
    r2 = 2 * (1 + np.cos(2 * np.arccos(alpha) / 3))
    b1 = - (r1 ** 3 - 3 * r1 ** 2 + alpha ** 2 * r1 + alpha ** 2) / (alpha * (r1 - 1))
    b2 = - (r2 ** 3 - 3 * r2 ** 2 + alpha ** 2 * r2 + alpha ** 2) / (alpha * (r2 - 1))
    return b1, b2, max(abs(r1), abs(r2))


#################################### DORMAND-PRINCE, uwu ###############################################################

# func(t, qp, *params)
# init = (q0, p0)
def RK45(func: Callable, init: Union[np.ndarray, List], t_interval: Union[List, tuple], h_init: float, BH: bool,
         r_plus: float, **params) -> tuple:

    if len(params) > (len(inspect.signature(func).parameters)-2):       # -2 за компенсиране на t, y
        warnings.warn("The number of parameters given exceeds the number of positional arguments. "
                      "The excess will be removed.")
        params = {key: value for key, value in params.items() if key in inspect.signature(func).parameters.keys()}
        if len(params) < (len(inspect.signature(func).parameters)-2):
            raise TypeError("Check your parameter names. Something is wrong.")

    # щото ми омръзна от input variables
    alpha = np.sqrt(1 - (r_plus-1) ** 2)
    b1, b2, r = b_values(alpha)

    rev = bool((t_interval[0] > t_interval[-1]) or (t_interval[0] < 0))

    # не е вярно
    if rev:
        t_interval = sorted(list(t_interval))       # и да е било tuple, вече не е
    elif t_interval[0] == t_interval[-1]:
        warnings.warn("Check your t intervals.")

    t = np.zeros(shape=NUM_ITER)
    y = np.zeros(shape=(NUM_ITER, 2*len(init)))
    t[0] = t_interval[0]
    y[0] = np.tile(init, 2)
    h = h_init

    k = np.zeros(shape=(7, len(init)), dtype=float)

    current_r = list()
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
            current_r.append(y[j, 1])

            # не ми харесва колко if-ове има и повтарящ се код
            #if BH and (b1 < y[j, 7] < b2):     # radial component < 40
            #    if (y[0, 5] < 0 and rev) or (y[0, 5] > 0 and not rev):      # p_r
            #        y = y.transpose()[:len(init)]
            #        y = y[:, :np.count_nonzero(y[1])]      # remove zero elements
            #        return False, t, y      # not fallen
            #    elif y[j, 1] < r_plus + DELTA:       # avoiding circular imports, uwu(?)
            #        y = y.transpose()[:len(init)]
            #        y = y[:, :np.count_nonzero(y[1])]
            #        return True, t, y
            #elif BH and not (b1 < y[j, 7] < b2):
            #    if y[j, 1] < r:
            #        y = y.transpose()[:len(init)]
            #        y = y[:, np.count_nonzero(y[1])]
            #        return True, t, y
            if y[j, 1] < r_plus + DELTA:
                y = y.transpose()[:len(init)]
                y = y[:, :np.count_nonzero(y[1])]
                return True, t, y

            j += 1
        else:
            # като няма достатъчно точност искаме по-добра стъпка, не дубликати на предните резултати???????
            # t[j] = t[j-1]
            # y[j, :] = y[j-1, :]
            h *= 0.8 * (RK45_ACCURACY / (err.cur_err + 1e-10)) ** 0.25

    y = y.transpose()[:len(init)]
    return 0, t, y
