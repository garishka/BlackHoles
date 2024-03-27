import numpy as np
from typing import Union

from zipoy_voorhees_spacetime.gamma_metric import g_cov, gamma_infty_metric
from zipoy_voorhees_spacetime.geodesics import GammaMimicker


class Observer(GammaMimicker):

    def __init__(self, position=np.array([500, np.pi / 2]), gamma=0.51):

        super().__init__(gamma)
        self.position = position

    def perim_r(self) -> float:
        """
        Return the perimetral radius.
        """
        if self.gamma == np.inf:
            return np.sqrt(gamma_infty_metric(*self.position)[3, 3])
        return np.sqrt(g_cov(*self.position, self.gamma)[3, 3])

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
        if self.gamma == np.inf:
            g = gamma_infty_metric(*self.position)
        else:
            g = g_cov(*self.position, self.gamma)

        sin_i = np.sin(i)
        cos_i = np.cos(i)

        qp0 = np.zeros(shape=(8, ), dtype=float)

        # perhaps
        qp0[1] = np.sqrt((d ** 2 + alpha ** 2 + beta ** 2))
        qp0[2] = np.arccos((d * cos_i + beta * sin_i) / qp0[1])
        qp0[3] = np.arctan(alpha / (d * sin_i - beta * cos_i))

        qp0[5] = d / qp0[1]
        qp0[6] = (- cos_i + d * (d * cos_i + beta * sin_i) / qp0[1] ** 2) / np.sqrt(qp0[1] ** 2 - (d * cos_i + beta * sin_i) ** 2)
        qp0[7] = (- alpha * sin_i) / (alpha ** 2 + (d * cos_i + beta * sin_i) ** 2)
        qp0[4] = - np.sqrt(- g[1, 1] * qp0[5] ** 2 - g[2, 2] * qp0[6] ** 2 + g[3, 3] * qp0[7] ** 2)

        return qp0

    def gamma_g(self) -> float:
        if self.gamma == np.inf:
            g = gamma_infty_metric(*self.position)
        else:
            g = g_cov(*self.position, self.gamma)
        return -g[0, 3] / np.sqrt(g[3, 3] * (g[0, 3] ** 2 - g[0, 0] * g[3, 3]))

    def zeta(self) -> float:
        if self.gamma == np.inf:
            g = gamma_infty_metric(*self.position)
        else:
            g = g_cov(*self.position, self.gamma)
        return np.sqrt(g[3, 3] / (g[0, 3] ** 2 - g[0, 0] * g[3, 3]))

    def qp_zamo(self, alpha: float, beta: float) -> list[float]:
        if self.gamma == np.inf:
            g = gamma_infty_metric(*self.position)
        else:
            g = g_cov(*self.position, self.gamma)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)
        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)

        qp0 = np.zeros(shape=(8,), dtype=float)

        qp0[1] = self.position[0]
        qp0[2] = self.position[1]

        qp0[4] = (1 + self.gamma_g() * np.sqrt(g[3, 3]) * sin_b * cos_a) / self.zeta()
        qp0[5] = np.sqrt(g[1, 1]) * cos_a * cos_b
        qp0[6] = np.sqrt(g[2, 2]) * sin_a
        qp0[7] = np.sqrt(g[3, 3]) * sin_b * cos_a

        return qp0

# obs = Observer()
# print(obs.qp_init(np.pi/3, np.pi/4))
# [ 0.00000000e+00  2.50001713e+05  1.57079319e+00  2.09439204e-03
#  -3.04986929e+02  1.99998629e-03  2.51322242e-14 -6.11154981e-01]
