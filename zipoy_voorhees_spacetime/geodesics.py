# c = G = M = 1
import numpy as np

from zipoy_voorhees_spacetime.gamma_metric import g_cov, A_expr, B_expr, C_expr, gamma_infty_metric


class GammaMimicker:

    def __init__(self, gamma=0.51):
        self.gamma = gamma

    def __str__(self) -> str:
        return f"Gamma Metric Mimicker(alpha={self.gamma})"

    def r_isco(self) -> float:
        if self.gamma == np.infty:
            return 3 + np.sqrt(5)
        return 1 / self.gamma + 3 + np.sqrt(5 - 1 / self.gamma ** 2)

    def r_photon_capture(self) -> float:
        if self.gamma == np.inf:
            return 2
        return 2 + 1 / self.gamma

    def r_inf_redshift(self) -> float:
        if self.gamma == np.inf:
            return 0.
        return 2 / self.gamma

    def eff_potential_photons(self, r, theta, Lz):
        if self.gamma == np.inf:
            return "не ми се пише"
        return (Lz / r / np.sin(theta)) ** 2 * (1 - 2 / r / self.gamma) ** (2 * self.gamma - 1)


class GammaGeodesics:
    def __init__(self, gamma=0.51, null=True):
        self.gamma = gamma
        self.null = null

    def metric_r_deriv(self, r, theta):
        gamma = self.gamma
        #A = A_expr(r, gamma)
        #B = B_expr(r, theta, gamma)
        #C = C_expr(r, theta, gamma)
        #B_upper = r ** 2 - 2 * r / gamma
        #B_lower = r ** 2 - 2 * r / gamma + (np.sin(theta) / gamma) ** 2

        #dAdr = 2 * (1 - 2 / gamma / r) ** (gamma - 1) / r ** 2
        #dBudr = dBldr = dDdr = 2 * (r - 1 / gamma)
        #dBdr = (gamma ** 2 - 1) * (B_upper / B_lower) ** (gamma ** 2 - 2) * dBudr * (1 - B_upper / B_lower) / B_lower
        #dCdr = gamma ** 2 * (B_upper / B_lower) ** (gamma ** 2 - 1) * dBudr - (gamma ** 2 - 1) * (B_upper / B_lower) ** (gamma ** 2) * dBldr

        dgdr = np.zeros(shape=(4, ), dtype=float)

        #dgdr[0] = - dAdr
        #dgdr[1] = dBdr / A - B * dAdr / A ** 2
        #dgdr[2] = dCdr / A - C * dAdr / A ** 2
        #dgdr[3] = dDdr / A - B_upper * np.sin(theta) ** 2 * dAdr / A ** 2

        dgdr[0] = -2*(1 - 2/(gamma*r))**gamma/(r**2*(1 - 2/(gamma*r)))
        dgdr[1] = (((r**2 - 2*r/gamma)/(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2))**(gamma**2 - 1)*(gamma**2 - 1)*
                   ((-2*r + 2/gamma)*(r**2 - 2*r/gamma)/(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)**2 + (2*r - 2/gamma)
                   /(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2))*(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)/((1 - 2/
                   (gamma*r))**gamma*(r**2 - 2*r/gamma)) - 2*((r**2 - 2*r/gamma)/(r**2 - 2*r/gamma +
                   np.sin(theta)**2/gamma**2))**(gamma**2 - 1)/(r**2*(1 - 2/(gamma*r))*(1 - 2/(gamma*r))**gamma))
        dgdr[2] = (gamma**2*(2*r - 2/gamma)*(r**2 - 2*r/gamma)**(gamma**2)*(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)**
                   (1 - gamma**2)/((1 - 2/(gamma*r))**gamma*(r**2 - 2*r/gamma)) + (1 - gamma**2)*(2*r - 2/gamma)*
                   (r**2 - 2*r/gamma)**(gamma**2)*(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)**(1 - gamma**2)/
                   ((1 - 2/(gamma*r))**gamma*(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)) - 2*(r**2 - 2*r/gamma)**(gamma**2)
                   *(r**2 - 2*r/gamma + np.sin(theta)**2/gamma**2)**(1 - gamma**2)/(r**2*(1 - 2/(gamma*r))*(1 - 2/(gamma*r))**gamma))
        dgdr[3] = (2*r - 2/gamma)*np.sin(theta)**2/(1 - 2/(gamma*r))**gamma - 2*(r**2 - 2*r/gamma)*np.sin(theta)**2/(r**2*(1 - 2/(gamma*r))*(1 - 2/(gamma*r))**gamma)


        return dgdr

    def metric_th_deriv(self, r, theta):
        gamma = self.gamma
        A = A_expr(r, gamma)
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        B_upper = r ** 2 - 2 * r / gamma
        B_lower = r ** 2 - 2 * r / gamma + (sin_th / gamma) ** 2
        dBldth = 2 * sin_th * cos_th / gamma ** 2

        dgdth = np.zeros(shape=(4,), dtype=float)

        dgdth[2] = - (gamma ** 2 - 1) * (B_upper / B_lower) ** (gamma ** 2) * dBldth / A
        dgdth[1] = dgdth[2] / B_upper
        dgdth[3] = (r ** 2 - 2 * r / gamma) * 2 * sin_th * cos_th / A

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

        dzdl[0] = E / g[0, 0]
        dzdl[1] = p_r / g[1, 1]
        dzdl[2] = p_th / g[2, 2]
        dzdl[3] = L / g[3, 3]

        dzdl[4] = 1e-16  # trying if a very small number ≠0 would be better for numerical computations
        dzdl[5] = 0.5 * ((E / g[0, 0]) ** 2 * dgdr[0] + (p_r / g[1, 1]) ** 2 * dgdr[1] + (p_th / g[2, 2]) ** 2 *
                          dgdr[2] + (L / g[3, 3]) ** 2 * dgdr[3])
        dzdl[6] = 0.5 * ((p_r / g[1, 1]) ** 2 * dgdth[1] + (p_th / g[2, 2]) ** 2 * dgdth[2] + (L / g[3, 3]) ** 2 * dgdth[3])
        dzdl[7] = 1e-16

        return dzdl


class GammaInftyGeodesics:
    def __init__(self, null=True):
        self.null = null

    def metric_r_deriv(self, r, theta):
        a = np.exp(-2/r)
        a_inv = np.exp(2/r)
        sin_th = np.sin(theta)

        dgdr = np.zeros(shape=(4,), dtype=float)

        dgdr[0] = - 2 * a / r ** 2
        dgdr[1] = (-2 * a_inv * np.exp(- sin_th ** 2 / r ** 2) / r ** 2 + 2 * a_inv * np.exp(- sin_th ** 2 / r ** 2) * sin_th ** 2 / r ** 3)
        dgdr[2] = 2 * r * a_inv - 2 * a_inv
        dgdr[3] = 2 * r * a_inv * sin_th ** 2 - 2 * a_inv * sin_th ** 2

        return dgdr

    def metric_th_deriv(self, r, theta):
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        a_inv = np.exp(2/r)

        dgdth = np.zeros(shape=(4,), dtype=float)

        dgdth[1] = -2 * a_inv * np.exp(-sin_th ** 2 / r ** 2) * sin_th * cos_th / r ** 2
        dgdth[3] = 2 * r ** 2 * a_inv * sin_th * cos_th

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
        g = gamma_infty_metric(r, th)
        dgdr = self.metric_r_deriv(r, th)
        dgdth = self.metric_th_deriv(r, th)

        dzdl = np.zeros(shape=8, dtype=float)

        dzdl[0] = E / g[0, 0]
        dzdl[1] = p_r / g[1, 1]
        dzdl[2] = p_th / g[2, 2]
        dzdl[3] = L / g[3, 3]

        dzdl[4] = 1e-16  # trying if a very small number ≠0 would be better for numerical computations
        dzdl[5] = 0.5 * ((E / g[0, 0]) ** 2 * dgdr[0] + (p_r / g[1, 1]) ** 2 * dgdr[1] + (p_th / g[2, 2]) ** 2 *
                         dgdr[2] + (L / g[3, 3]) ** 2 * dgdr[3])
        dzdl[6] = 0.5 * ((p_r / g[1, 1]) ** 2 * dgdth[1] + (p_th / g[2, 2]) ** 2 * dgdth[2] + (L / g[3, 3]) ** 2 * dgdth[3])
        dzdl[7] = 1e-16

        return dzdl
