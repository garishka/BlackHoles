from typing import Union, List
import numpy as np


def discretize_angles(angles: Union[np.ndarray, list, float], resolution: int, interval: Union[float, np.ndarray, list]):
    """
    Discretize a list of angle values based on the specified resolution and interval.

    Parameters:
    -----------
    angles : list
        A list of angle values to be discretized.
    resolution : int
        The resolution for discretization, representing the number of steps per interval.
    interval : float, optional
        The domain length of the angle variable. Defaults to π.

    Returns:
    --------
    tuple
        A tuple containing the discretized angles and the uncertainty associated with them.

    Notes:
    ------
    The function discretizes each angle in the input list to the nearest multiple of
    the angular step size determined by the resolution and interval.

    Examples:
    ---------
    >>> angles = [0.5, 1.5, 2.5]
    >>> resolution = 10
    >>> interval = np.pi
    >>> discretize_angles(angles, resolution, interval)
    (array([0.62831853, 1.57079633, 2.51327412]), 0.3141592653589793)
    """
    if type(interval) == float:
        delta = interval / resolution
    else:
        delta = (interval[-1]-interval[0])/resolution

    angles = np.array(angles)
    disc_angles = np.round(angles / delta) * delta

    return disc_angles, delta


def map_angle_to_pixel(theta: float, phi: float, resolution: int, intervals: Union[np.ndarray, List]):
    """
    Map spherical coordinates (θ, ϕ) to pixel coordinates on an image.

    Parameters:
    -----------
    theta : float
        θ coordinate value, representing the polar angle.
    phi : float
        ϕ coordinate value, representing the azimuthal angle.
    resolution : int
        Resolution of the image in pixels.

    Returns:
    --------
    list
        A list of image coordinates corresponding to the given spherical coordinates (θ, ϕ).
    """
    # Generate a grid for the celestial sphere
    theta_values = np.linspace(intervals[0][0], intervals[0][1], resolution)
    phi_values = np.linspace(intervals[1][0], intervals[1][1], resolution)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

    # Adjust θ if it's negative
    if theta < 0:
        theta = 2 * np.pi - theta

    # Discretize input angles
    discretized_theta, delta_theta = discretize_angles(theta, resolution, intervals[0])
    discretized_phi, delta_phi = discretize_angles(phi, resolution, interval=intervals[1])

    # Find indices where the discretized angles match the meshgrid values
    indices = np.column_stack(np.where(np.isclose(theta_grid, discretized_theta, atol=delta_theta) &
                                       np.isclose(phi_grid, discretized_phi, atol=delta_phi)))

    # Choose the point closest to the center of the image
    if len(indices) > 1:
        center = np.array([resolution // 2, resolution // 2])
        distances = np.linalg.norm(indices - center, axis=1)
        closest_index = indices[np.argmin(distances)]
    else:
        closest_index = indices

    # for debugging
    # print("Input Coordinates:", theta, phi)
    # print("Delta theta:", delta_theta)
    # print("Delta Phi:", delta_phi)
    # print("Discretized Coordinates:", discretized_theta, discretized_phi)
    # print("Grid Points:")
    # print("Theta:", theta_values)
    # print("Phi:", phi_values)

    return closest_index.tolist()


def map_celestial(angles: tuple, obs_r: float, cs_r: float):
    # Cunha, p.121
    th, phi = angles
    r = np.sqrt(obs_r ** 2 + cs_r ** 2 - 2 * obs_r * cs_r * np.sin(th) * np.cos(phi))

    alpha = np.arcsin(cs_r * np.cos(th) / r)
    a = - (cs_r * np.sin(th) * np.sin(phi)) / (r * np.cos(alpha))
    b = (obs_r - cs_r * np.sin(th) * np.cos(phi)) / (r * np.cos(alpha))

    if b >= 0:
        beta = np.arcsin(a)
    elif (b < 0) and (a >= 0):
        beta = np.pi - np.arcsin(a)
    else:
        beta = -np.pi - np.arcsin(a)

    return alpha, beta
