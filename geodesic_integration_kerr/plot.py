import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import geodesics
from scipy.integrate import solve_ivp
from PIL import Image, ImageDraw

# image resolution
res = 50

preamble = [r'\usepackage[utf8]{inputenc}',
    r'\usepackage[bulgarian]{babel}',
    r"\usepackage{amsmath}",
    r'\usepackage{siunitx}',
    r'\usepackage{emoji}']

mpl.use("pgf")
LaTeX = {
"text.usetex": True,
"font.family": "CMU Serif",
    "pgf.preamble": "\n".join(line for line in preamble),
    "pgf.rcfonts": True,
    "pgf.texsystem": "lualatex"}
plt.rcParams.update(LaTeX)


def discretize_angles(angles: list, resolution: int, interval=np.pi):
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
    delta = interval / resolution
    angles = np.array(angles)

    disc_angles = np.round(angles / delta) * delta

    return disc_angles, delta


def map_angle_to_pixel(theta: float, phi: float, resolution: int):
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
    theta_values = np.linspace(0, np.pi, resolution)
    phi_values = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

    # Adjust θ if it's negative
    if theta < 0:
        theta = np.pi - theta

    # Discretize input angles
    discretized_theta, delta_theta = discretize_angles([theta], resolution)
    discretized_phi, delta_phi = discretize_angles([phi], resolution, interval=2 * np.pi)

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
    print("Input Coordinates:", theta, phi)
    print("Discretized Coordinates:", discretized_theta, discretized_phi)
    print("Grid Points:")
    print("Theta:", theta_values)
    print("Phi:", phi_values)

    return closest_index.tolist()


# Generate a grid of angle values (β, γ) for the observer's viewpoint
# This grid might be adjusted for small angles since the entire calculation for x and y relies on this.
beta_values = np.linspace(np.pi/2, -np.pi/2, res)
gamma_values = np.linspace(np.pi/2, -np.pi/2, res)
beta, gamma = np.meshgrid(beta_values, gamma_values)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = geodesics.KerrBlackHole(alpha=0.99)
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (15, π, 0), a.k.a default position
obs = geodesics.Observer()
init_q = obs.coord()

# Create an empty white image with a resolution of res
image = Image.new("RGB", (res, res), "white")
draw = ImageDraw.Draw(image)
pixels = image.load()

# Load an image used for the celestial sphere background
background = Image.open("background/patterned_circles.png")
px_bg = background.load()

counter = 0
# Loop through each (β, γ) coordinate on the observer's plane
for i in range(len(beta)):
    for j in range(len(gamma)):
        # Solve the geodesic equations for the current (β, γ) with given initial conditions
        init_p = obs.p_init(gamma[i, j], beta[i, j])
        geo = geodesics.Geodesics([0.99], init_q, init_p)
        ivp = np.array([0, init_q[0], init_q[1], init_q[2], init_p[1], init_p[2]])
        sol = solve_ivp(geo.hamilton_eqs, [0, -30], ivp, t_eval=np.linspace(0, -30, 5000))

        # Temporary print statement for debugging
        print(i, j)

        # Check if the light ray falls into the black hole; if not, map it to the celestial sphere
        for k in range(len(sol.y[1])):
            # Need to estimate the error for r
            # TODO: да поправя условието за падане в дупката
            if abs(sol.y[1, k]) <= r_plus + 1e-1:
                # The light ray falls into the black hole; set the pixel to black
                pixels[i, j] = (0, 0, 0)
                print("padna v dupkata")
                break
            try:
                if abs(abs(sol.y[1, k]) - 30) < 1e-1:
                    # Light ray hits the celestial sphere; map it to a pixel on the background image
                    coord = np.asarray([sol.y[2, k], sol.y[3, k]]) / np.pi
                    print(coord)
                    px_coord = map_angle_to_pixel(coord[0], coord[1], res)
                    pixels[i, j] = px_bg[px_coord[0], px_coord[1]]
                    # Used for debugging
                    counter += 1
                    break
            except IndexError:
                pixels[i, j] = (0, 0, 0)
                print("retard")


image.save("test_hole.png")

# TODO: да свържа пикселите с импакт параметрите
# TODO: да начертая графиката

# това е да видя каква част от пикселите са минали условието за падане
print(counter)
