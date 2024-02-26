# c = G = M = 1
import numpy as np


class GammaMimicker:

    def __init__(self, gamma=0.51):
        self.gamma = gamma

    def __str__(self) -> str:
        return f"Gamma Metric Mimicker(alpha={self.gamma})"

    def r_isco(self):
        return 1 / self.gamma + 3 + np.sqrt(5 - 1 / self.gamma ** 2)

    def r_photon_capture(self) -> float:
        return 1 + 1 / self.gamma

    def eff_potential_photons(self, r, theta, Lz):
        return (Lz / r / np.sin(theta)) ** 2 * (1 - 2 / r / self.gamma) ** ( 2 * self.gamma - 1)

