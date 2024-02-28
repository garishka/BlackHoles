# c = G = M = 1
import numpy as np

from zipoy_voorhees_spacetime.gamma_metric import g_cov, A_expr, B_expr, C_expr


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
        return (Lz / r / np.sin(theta)) ** 2 * (1 - 2 / r / self.gamma) ** (2 * self.gamma - 1)


class GammaGeodesics:
    def __init__(self, gamma=0.51, null=True):
        self.gamma = gamma
        self.null = null

    def metric_r_deriv(self, r, theta):
        A = A_expr(r, self.gamma)
        C = C_expr(r, theta, self.gamma)
        B_upper = r ** 2 - 2 * r / self.gamma
        B_lower = r ** 2 - 2 * r / self.gamma + (np.sin(theta) / self.gamma) ** 2

        dAdr = 2 * (1 - 2 / self.gamma / r) ** (self.gamma - 1) / r ** 2
        dBudr = dBldr = dDdr = 2 * (r - 1 / self.gamma)
        dBdr = (self.gamma ** 2 - 1) * ((B_upper / B_lower) ** (self.gamma ** 2 - 1) / B_upper * dBudr -
                (B_upper / B_lower) ** (self.gamma ** 2) / B_upper * dBldr)
        dCdr = self.gamma ** 2 * (B_upper / B_lower) ** (self.gamma ** 2) * (B_lower / B_upper * dBudr - dBldr)

        dgdr = np.zeros(shape=(4, ), dtype=float)

        dgdr[0] = - dAdr
        dgdr[1] = dCdr / A - C * dAdr / A ** 2
        dgdr[2] = dCdr / A - C * dAdr / A ** 2
        dgdr[3] = dDdr / A - B_upper * dAdr / A ** 2

        return dgdr

    def metric_th_deriv(self, r, theta):
        A = A_expr(r, self.gamma)
        B_upper = r ** 2 - 2 * r / self.gamma
        B_lower = r ** 2 - 2 * r / self.gamma + (np.sin(theta) / self.gamma) ** 2

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        dBldth = 2 * sin_th * cos_th / self.gamma ** 2

        dgdth = np.zeros(shape=(4,), dtype=float)

        dgdth[2] = (self.gamma ** 2 - 1) * (B_upper / B_lower) ** (self.gamma ** 2) / A * dBldth
        dgdth[1] = dgdth[2] / B_upper
        dgdth[3] = (r ** 2 - 2 * r / self.gamma) * 2 * sin_th * cos_th / A

        return dgdth

    def hamilton_eqs(self, l: float, qp: np.ndarray) -> np.ndarray:
        """
        Calculate and return the Hamiltonian equations for the geodesic motion.

        Parameters:
        -----------
        l: float
            Affine parameter.
        qp: np.ndarray
            Array of positions and momenta.
            (t, r, θ, ϕ, p_t, p_r, p_θ, p_ϕ)

        Returns:
        -----------
        list[float]
            List of Hamiltonian equations for the geodesic motion.
            The equations represent the evolution of the geodesic coordinates and momenta over the affine parameter l.
        """
        t, r, th, phi = qp[:4]
        E, p_r, p_th, L = qp[4:]
        gamma = self.gamma
        g = g_cov(r, th, gamma)
        dgdr = self.metric_r_deriv(r, th)
        dgdth = self.metric_th_deriv(r, th)

        dzdl = np.zeros(shape=8, dtype=float)

        dzdl[0] = - E / g[0, 0]
        dzdl[1] = p_r / g[1, 1]
        dzdl[2] = p_th / g[2, 2]
        dzdl[3] = L / g[3, 3]

        dzdl[4] = 1e-15  # trying if a very small number ≠0 would be better for numerical computations
        dzdl[5] = - 0.5 * (- (E / g[0, 0]) ** 2 * dgdr[0] + (p_r / g[1, 1]) ** 2 * dgdr[2] * (p_th / g[2, 2]) ** 2 *
                           dgdr[2] * (L / g[3, 3]) ** 2 * dgdr[3])
        dzdl[6] = - 0.5 * ((p_r / g[1, 1]) ** 2 * dgdth[2] * (p_th / g[2, 2]) ** 2 *
                           dgdth[2] * (L / g[3, 3]) ** 2 * dgdth[3])
        dzdl[7] = 1e-15

        return dzdl
