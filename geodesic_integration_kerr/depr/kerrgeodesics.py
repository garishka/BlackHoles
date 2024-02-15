# c = G = M = 1
import numpy as np
from geodesic_integration_kerr.depr import metricKerr


class KerrBlackHole:
    """
    Represents a Kerr black hole with a specified spin parameter.

    Parameters:
    -----------
    alpha : float
        Spin parameter of the Kerr black hole. Must be in the open interval (-1, 1).

    Methods:
    -----------
    __str__():
        Return a string representation of the KerrBlackHole instance.

    r_plus() -> float:
        Return the radius of the event horizon for the Kerr black hole with the given spin parameter.
    """
    def __init__(self, alpha=0.99):
        """
                Initialize a KerrBlackHole instance with the specified spin parameter.

                Parameters:
                -----------
                alpha : float
                    Spin parameter of the Kerr black hole. Must be in the open interval (-1, 1).
                """
        self.alpha = alpha

    def __str__(self) -> str:
        return f"KerrBlackHole(alpha={self.alpha})"

    def r_plus(self) -> float:
        """
        Return the radius of the event horizon for the Kerr black hole with the given spin parameter.
        """
        return 1 + np.sqrt(1 - self.alpha ** 2)


class KerrGeodesics(KerrBlackHole):
    def __init__(self, alpha=0.99, null=True):
        """
        Constructor

        Parameters:
        ------------
        metric_params: array-like
            Tuple of parameters to pass to the metric.
            param[0] = alpha, the spin parameter of the black hole.
        position: array-like
            4-position[1:3] where (t=0, ϕ=0) of the geodesic observer in the Kerr metric.
        momentum: array-like
            4-momentum of the geodesic observer.
        null: bool
            Default null=True, whether the geodesic is null.
            Currently useless, UwU
        """
        super().__init__(alpha)
        self.null = null

    def dgdtheta(self, r, th):
        # Calculate and return the derivative of the contravariant Kerr metric with respect to the polar angle (θ)
        alpha = self.alpha
        sigma = metricKerr.sigma_expr(r, th, alpha)
        A = metricKerr.A_expr(r, th, alpha)
        delta = metricKerr.delta_expr(r, alpha)

        dgdth = np.zeros(shape=(4, 4), dtype=float)

        dgdth[0, 0] = (delta * sigma - A) * np.sin(2 * th) * alpha ** 2 / (delta * sigma ** 2)
        dgdth[0, 3] = dgdth[3, 0] = - 2 * r * np.sin(2 * th) * alpha ** 3 / (delta * sigma ** 2)
        dgdth[1, 1] = alpha ** 2 * np.sin(2 * th) * delta / sigma ** 2
        dgdth[2, 2] = alpha ** 2 * np.sin(2 * th) / sigma ** 2
        dgdth[3, 3] = - np.sin(2 * th) * (alpha ** 2 + (delta - (alpha * np.sin(th)) ** 2 * (sigma
                      - (alpha * np.sin(th)) ** 2) / (sigma * np.sin(th)))) / (delta * sigma * np.sin(th) ** 2)

        return dgdth

    def dgdr(self, r, th):
        # Calculate and return the derivative of the contravariant Kerr metric with respect to the radial coordinate (r)
        alpha = self.alpha
        sigma = metricKerr.sigma_expr(r, th, alpha)
        A = metricKerr.A_expr(r, th, alpha)
        delta = metricKerr.delta_expr(r, alpha)

        dgdr = np.zeros(shape=(4, 4), dtype=float)

        dAdr = 4 * r * (r ** 2 + alpha ** 2) - 2 * (r - 1) * alpha ** 2 * np.sin(th) ** 2

        dgdr[0, 0] = - dAdr / (sigma * delta) + 2 * A * (r * (sigma + delta) - sigma) / (sigma * delta) ** 2
        dgdr[0, 3] = dgdr[3, 0] = - 2 * alpha / (sigma * delta) + 4 * r * alpha * (r * (sigma + delta) - sigma) / (
                     sigma * delta) ** 2
        dgdr[1, 1] = 2 * r * (sigma - delta) / sigma ** 2 - 2 / sigma
        dgdr[2, 2] = - 2 * r / sigma ** 2
        dgdr[3, 3] = (2 * (r - 1) / (delta * sigma) - 2 * (delta - (alpha * np.sin(th)) ** 2) * (r * (sigma + delta) -
                      sigma) / (delta * sigma) ** 2) / np.sin(th) ** 2

        return dgdr

    # TODO: да добавя начин за проверка на запазването на енергията и момента на импулса
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
        alpha = self.alpha
        g = metricKerr.contra_Kerr_metric(r, th, alpha)
        dgdth = self.dgdtheta(r, th)
        dgdr = self.dgdr(r, th)

        dzdl = np.zeros(shape=8, dtype=float)

        dzdl[0] = g[0, 0] * E + g[0, 3] * L
        dzdl[1] = g[1, 1] * p_r
        dzdl[2] = g[2, 2] * p_th
        dzdl[3] = g[0, 3] * E + g[3, 3] * L

        dzdl[4] = 1e-15  # trying if a very small number ≠0 would be better for numerical computations
        dzdl[5] = - 0.5 * (dgdr[0, 0] * E ** 2 + 2 * dgdr[0, 3] * E * L + dgdr[1, 1] * p_r ** 2 + dgdr[2, 2] * p_th ** 2
                           + dgdr[3, 3] * L ** 2)
        dzdl[6] = - 0.5 * (dgdth[0, 0] * E ** 2 + 2 * dgdth[0, 3] * E * L + dgdth[1, 1] * p_r ** 2 + dgdth[2, 2] * p_th ** 2
                           + dgdth[3, 3] * L ** 2)
        dzdl[7] = 1e-15

        return dzdl

# g = Kerr_metric([15, np.pi/2, 0], 0.99)
# print(np.linalg.det(g))
# analytic = g[0, 0] * g[1, 1] * g[2, 2] * g[3, 3] - g[0, 3] ** 2 * g[1, 1] * g[2, 2]
# print(analytic)
# => дават еднакъв резултат
