import numpy as np


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
        position in the form (r, θ).
    params : array-like, optional
        List of parameters. By default, params[0]=α, the spin parameter.

    Returns:
    -----------
    np.ndarray
        A (4, 4) array representing the covariant metric components.
    """
    r, th = x_vec
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
        position in the form (r, θ).
    params : array-like, optional
        List of parameters. By default, params[0]=α, the spin parameter.

    Returns:
    -----------
    np.ndarray
        A (4, 4) array representing the contravariant metric components in Boyer-Lindquist coordinates.
    """
    #r, th = x_vec
    #alpha = params[0]

    #sigma = sigma_expr(r, th, alpha)
    #delta = delta_expr(r, alpha)
    #A = A_expr(r, th, alpha)

    #g = np.zeros(shape=(4, 4), dtype=float)

    #g[0, 0] = - A / (delta * sigma)
    #g[0, 3] = g[3, 0] = - 2 * r * alpha / (delta * sigma)
    #g[3, 3] = (delta - (alpha * np.sin(th) ** 2)) / (delta * sigma * np.sin(th) ** 2)
    #g[1, 1] = delta / sigma
    #g[2, 2] = 1 / sigma

    return np.linalg.inv(Kerr_metric(x_vec, *params))
