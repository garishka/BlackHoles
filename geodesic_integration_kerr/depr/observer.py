import numpy as np
from typing import Union

from metricKerr import Kerr_metric


# не ми харесва как е дефиниран тоя клас
class Observer:
    """
    Represents an observer in the vicinity of a Kerr black hole.

    Parameters:
    -----------
    position : array-like, optional
        3-position of the observer. Defaults to [15, π/2, 0].
    alpha : float, optional
        Spin parameter of the black hole. Defaults to 0.99.

    Methods:
    -----------
    coord() -> array-like:
        Return the 3-position of the observer.

    perim_r() -> float:
        Return the perimetral radius based on the Kerr metric.

    impact_params(a1: float, a2: float) -> Tuple[float, float]:
        Calculate and return the impact parameters (x, y) of the light ray reaching the observer.

    gamma_g() -> float:
        Calculate and return the gamma factor for the observer.

    zeta() -> float:
        Calculate and return the zeta factor for the observer.

    p_init(a1: float, a2: float) -> List[float]:
        Calculate and return the initial conditions for the 4-momentum for a given set of angles.

    Notes:
    -----------
    The Observer class provides methods for calculating various parameters and conditions
    relevant to an observer near a rotating black hole, based on the Kerr metric in Boyer-Lindquist coordinates.
    """

    def __init__(self, position=np.array([500, np.pi / 2]), alpha=0.99):
        """
        Constructor

        Parameters
        ----------
        :param position: array-like, optional
            2-position of the observer. Defaults to [15, π/2].
        :param alpha: float, optional
            Spin parameter of the black hole. Defaults to 0.99.
        """

        #if position is []:
        #    position =
        self.position = position
        self.alpha = alpha

    def coord(self) -> np.ndarray:
        """
        Return the 3-position of the observer.
        """
        return np.array(self.position)

    def perim_r(self) -> float:
        """
        Return the perimetral radius based on the Kerr metric.
        """
        return np.sqrt(Kerr_metric(self.position, self.alpha)[3, 3])

    def impact_params(self, a1: Union[float, np.ndarray], a2: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate and return the impact parameters (x, y) of the light ray reaching the observer.

        Parameters:
        ----------
        a1: float or np.ndarray
            The angle(s) associated with the y-axis.
        a2: float or np.ndarray
            The angle(s) associated with the x-axis.

        Returns:
        ----------
        np.ndarray
            Impact parameters (x, y).
        """
        r_tilde = self.perim_r()

        if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
            x = -r_tilde * a2
            y = r_tilde * a1
        else:
            x = -r_tilde * np.array([a2])
            y = r_tilde * np.array([a1])

        return np.array([x, y])

    def gamma_g(self) -> float:
        metric = Kerr_metric(self.position, self.alpha)
        return -metric[0, 3] / np.sqrt(metric[3, 3] * (metric[0, 3] ** 2 - metric[0, 0] * metric[3, 3]))

    def zeta(self) -> float:
        metric = Kerr_metric(self.position, self.alpha)
        return np.sqrt(metric[3, 3] / (metric[0, 3] ** 2 - metric[0, 0] * metric[3, 3]))

    def p_init(self, a1: float, a2: float) -> list[float]:
        """
        Calculate and return the initial conditions for the 4-momentum for a given set of angles.

        Parameters:
        -----------
        a1: float
            The angle associated with the x-axis.
        a2: float
            The angle associated with the y-axis.

        Returns:
        -----------
        list[float]
            Initial conditions for the 4-momentum.
        Notes:
        ----------
        p_init[0] = E = const, p_init[3] = L = const
        """
        metric = Kerr_metric(self.position, self.alpha)

        p_th = np.sqrt(metric[2, 2]) * np.sin(a2)
        p_r = np.sqrt(metric[1, 1]) * np.cos(a1) * np.cos(a2)
        L = np.sqrt(metric[3, 3]) * np.sin(a1) * np.cos(a2)
        E = (1 + self.gamma_g() * np.sqrt(metric[3, 3]) * np.sin(a1) * np.cos(a2)) / self.zeta()

        return [E, p_r, p_th, L]
