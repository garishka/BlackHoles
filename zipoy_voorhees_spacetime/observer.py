import numpy as np
from typing import Union

from zipoy_voorhees_spacetime.gamma_metric import g_cov
from zipoy_voorhees_spacetime.geodesics import GammaMimicker


class Observer(GammaMimicker):

    def __init__(self, position=np.array([500, np.pi / 2]), gamma=0.51):

        super().__init__(gamma)
        self.position = position

    def coord(self) -> np.ndarray:
        """
        Return the 3-position of the observer.
        """
        if isinstance(self.position, np.ndarray):
            return self.position
        return np.array(self.position)

    def qp_init(self, alpha: float, beta: float) -> list[float]:
        # https://arxiv.org/pdf/1904.06207.pdf
        d, i = self.position
        g = g_cov(*self.position, self.gamma)

        sin_i = np.sin(i)
        cos_i = np.cos(i)

        qp0 = np.zeros(shape=(8, ), dtype=float)

        # perhaps
        qp0[1] = (d ** 2 + alpha ** 2 + beta ** 2)
        qp0[2] = np.arccos((d * cos_i + beta * sin_i) / qp0[1])
        qp0[3] = np.arctan(alpha / (d * sin_i - beta * cos_i))

        qp0[5] = d / qp0[1]
        qp0[6] = (- cos_i + d * (d * cos_i + beta * sin_i) / qp0[1] ** 2) / np.sqrt(qp0[1] ** 2 - (d * cos_i + beta * sin_i) ** 2)
        qp0[7] = (- alpha * sin_i) / (alpha ** 2 + (d * cos_i + beta * sin_i) ** 2)
        qp0[4] = - np.sqrt(- g[1, 1] * qp0[5] ** 2 - g[2, 2] * qp0[6] ** 2 + g[3, 3] * qp0[7] ** 2)

        return qp0

# obs = Observer()
# print(obs.qp_init(np.pi/3, np.pi/4))
# [ 0.00000000e+00  2.50001713e+05  1.57079319e+00  2.09439204e-03
#  -3.04986929e+02  1.99998629e-03  2.51322242e-14 -6.11154981e-01]
