# c = G = M = 1
import numpy as np
from typing import Union


def sigma_expr(r, theta, alpha):
    """
    Calculate the Σ expression from the Kerr metric in Boyer-Lindquist coordinates.

    Parameters:
    -----------
    r : float
        Radial coordinate in Boyer-Lindquist coordinates.
    theta : float
        θ angle in Boyer-Lindquist coordinates.
    alpha : float
        Spin parameter of the black hole. Must be in the open interval (-1, 1).

    Returns:
    -----------
    float
        The Σ expression representing a component of the Kerr metric in Boyer-Lindquist coordinates.
        Σ = r² + (α * cos(θ))²
    """
    return r ** 2 + (alpha * np.cos(theta)) ** 2


def delta_expr(r, alpha):
    """
    Calculate the Δ expression from the Kerr metric in Boyer-Lindquist coordinates.

    Parameters:
    -----------
    r : float
        Radial coordinate in Boyer-Lindquist coordinates.
    alpha : float
        Spin parameter of the black hole. Must be in the open interval (-1, 1).

    Returns:
    -----------
    float
        The Δ expression representing a component of the Kerr metric in Boyer-Lindquist coordinates.
        Δ = r² - 2r + α²
    """
    return r ** 2 - 2 * r + alpha ** 2


def A_expr(r, theta, alpha):
    """
        Calculate the A expression from the Kerr metric in Boyer-Lindquist coordinates.

        Parameters:
        -----------
        r : float
            Radial coordinate in Boyer-Lindquist coordinates.
        theta : float
            θ angle in Boyer-Lindquist coordinates.
        alpha : float
            Spin parameter of the black hole. Must be in the open interval (-1, 1).

        Returns:
        -----------
        float
            The A expression representing a component of the Kerr metric in Boyer-Lindquist coordinates.
            A = (r² + α²)² - Δ(r, θ)(α sin(θ))²
        """
    return (r ** 2 + alpha ** 2) ** 2 - delta_expr(r, alpha) * (alpha * np.sin(theta)) ** 2


def Kerr_metric(x_vec, *params):
    """
    Calculate the covariant form of the Kerr metric in Boyer-Lindquist coordinates.

    Parameters:
    -----------
    x_vec : array-like
        3-position in the form (r, θ, ϕ).
    params : array-like, optional
        List of parameters. By default, params[0]=α, the spin parameter.

    Returns:
    -----------
    np.ndarray
        A (4, 4) array representing the covariant metric components.
    """
    r, th, phi = x_vec
    alpha = params[0]

    sigma = sigma_expr(r, th, alpha)
    delta = delta_expr(r, alpha)
    A = A_expr(r, th, alpha)

    g = np.zeros(shape=(4, 4), dtype=float)

    g[0, 0] = 2 * r / sigma - 1
    g[0, 3] = g[3, 0] = -2 * r * alpha * np.sin(th) ** 2 / sigma
    g[3, 3] = A * np.sin(th) ** 2 / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma

    return g


def contra_Kerr_metric(x_vec, *params):
    """
    Calculate the contravariant form of the Kerr metric in Boyer-Lindquist coordinates.

    Parameters:
    -----------
    x_vec : array-like
        3-position in the form (r, θ, ϕ).
    params : array-like, optional
        List of parameters. By default, params[0]=α, the spin parameter.

    Returns:
    -----------
    np.ndarray
        A (4, 4) array representing the contravariant metric components.
    """
    r, th, phi = x_vec
    alpha = params[0]

    sigma = sigma_expr(r, th, alpha)
    delta = delta_expr(r, alpha)
    A = A_expr(r, th, alpha)

    g = np.zeros(shape=(4, 4), dtype=float)

    g[0, 0] = - A / (delta * sigma)
    g[0, 3] = g[3, 0] = - 2 * r * alpha / (delta * sigma)
    g[3, 3] = (delta - (alpha * np.sin(th) ** 2)) / (delta * sigma * np.sin(th) ** 2)
    g[1, 1] = delta / sigma
    g[2, 2] = 1 / sigma

    return g


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
    def __init__(self, alpha: float):
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

    def __init__(self, position=[15, np.pi / 2, 0], alpha=0.99):
        """
        Constructor

        Parameters
        ----------
        :param position: array-like, optional
            3-position of the observer. Defaults to [15, π/2, 0].
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


# TODO: да добавя възможност за времеподобни геодезични
class Geodesics:
    def __init__(self,  position, momentum, metric_params=[0.99], null=True):
        """
        Constructor

        Parameters
        ----------
        metric_params: array-like
            Tuple of parameters to pass to the metric.
            param[0] = alpha, the spin parameter of the black hole.
        position: array-like
            3-position (t=0) of the geodesic observer.
        momentum: array-like
            4-momentum of the geodesic observer.
        null: bool
            Default null=True, whether the geodesic is null.
            Currently useless, UwU
        """
        self.m_params = metric_params
        self.position = position
        self.momentum = momentum
        self.null = null

    # това най-вероятно ще стане ненужно с писането на integrator.py
    def dgdtheta(self):
        # Calculate and return the derivative of the geodesic components with respect to the polar angle (θ).
        alpha = self.m_params[0]
        r, th, phi = self.position
        sigma = sigma_expr(r, th, phi)
        A = A_expr(r, th, alpha)
        delta = delta_expr(r, alpha)

        dgdth = np.zeros(shape=(4, 4), dtype=float)

        dgdth[0, 0] = 2 * r * alpha ** 2 * np.sin(2 * th) / sigma ** 2
        dgdth[0, 3] = dgdth[3, 0] = - 4 * r * alpha * np.sin(2 * th) * (1 - (alpha * np.sin(th) ** 2) / sigma) / sigma
        dgdth[1, 1] = - alpha ** 2 * np.sin(2 * th) / delta
        dgdth[2, 2] = - alpha ** 2 * np.sin(2 * th)
        dgdth[3, 3] = np.sin(2 * th) * (2 * A - (r **2 + alpha ** 2) ** 2 + A * np.sin(th) ** 2 / sigma) / sigma

        return dgdth

    def dgdr(self):
        # Calculate and return the derivative of the geodesic components with respect to the radial coordinate (r).
        alpha = self.m_params[0]
        r, th, phi = self.position
        sigma = sigma_expr(r, th, phi)
        A = A_expr(r, th, alpha)
        delta = delta_expr(r, alpha)

        dgdr = np.zeros(shape=(4, 4), dtype=float)

        dAdr = 4 * r * (r ** 2 + alpha ** 2) - 2 * (r - 1) * alpha ** 2 * np.sin(th) ** 2

        dgdr[0, 0] = 2 * (1 - 2 * r ** 2 / sigma) / sigma
        dgdr[0, 3] = dgdr[3, 0] = - 2 * alpha * np.sin(th) ** 2 * (1 - 2 * r ** 2 / sigma) / sigma
        dgdr[1, 1] = 2 * (r + (1 - r) * sigma / delta) / delta
        dgdr[2, 2] = 2 * r
        dgdr[3, 3] = np.sin(th) ** 2 * (dAdr - 2 * r * A / sigma) / sigma

        return dgdr

    def hamiltonian(self):
        """
        Calculates the Hamiltonian for the geodesic motion.
        """
        alpha = self.m_params[0]
        p_t, p_r, p_th, p_phi = self.momentum
        g = contra_Kerr_metric(self.position, alpha)

        # за фотони
        tH = g[0, 0] * p_t ** 2 + g[1, 1] * p_r ** 2 + g[2, 2] * p_th ** 2 + g[0, 3] * p_phi * p_t + g[3, 3] * p_phi ** 2
        H = tH / 2

        return H

    # TODO: да добавя начин за проверка на запазването на енергията и момента на импулса
    # това дава грешни резултати с solve_ivp -> трябва да опитам нещо друго
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
        E, p_r, p_th, L = self.momentum
        alpha = self.m_params[0]
        g = Kerr_metric(self.position, alpha)
        dgdth = self.dgdtheta()
        dgdr = self.dgdr()

        dzdl = np.zeros(shape=(8), dtype=float)

        dzdl[0] = g[0, 0] * E + g[0, 3] * L
        dzdl[1] = g[1, 1] * p_r
        dzdl[2] = g[2, 2] * p_th
        dzdl[3] = g[0, 3] * E + g[3, 3] * L

        dzdl[4] = 0
        dzdl[5] = - 0.5 * (dgdr[0, 0] * E ** 2 + dgdr[0, 3] * E * L + dgdr[1, 1] * p_r ** 2 + dgdr[2, 2] * p_th ** 2
                           + dgdr[3, 3] * L ** 2)
        dzdl[6] = - 0.5 * (dgdth[0, 0] * E ** 2 + dgdth[0, 3] * E * L + dgdth[1, 1] * p_r ** 2 + dgdth[2, 2] * p_th ** 2
                           + dgdth[3, 3] * L ** 2)
        dzdl[7] = 0

        return dzdl


#g = Kerr_metric([15, np.pi/2, 0], 0.99)
#print(np.linalg.det(g))
#analytic = g[0, 0] * g[1, 1] * g[2, 2] * g[3, 3] - g[0, 3] ** 2 * g[1, 1] * g[2, 2]
#print(analytic)
# => дават еднакъв резултат

