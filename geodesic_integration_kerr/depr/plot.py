import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import geodesics
import metricKerr
from geodesic_integration_kerr.kdp_integrator import kdp45

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
beta_values = np.linspace((1-5e-3)*np.pi, (1+5e-3)*np.pi, res)
gamma_values = np.linspace(-5e-3*np.pi, 5e-3*np.pi, res)
beta, gamma = np.meshgrid(beta_values, gamma_values)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = metricKerr.KerrBlackHole(alpha=0.99)
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (15, π, 0), a.k.a default position
obs = metricKerr.Observer()
init_q = obs.coord()

# Impact parameters (np.ndarray), calculated at every angle (γ, β)
x, y = obs.impact_params(a1=gamma_values, a2=beta_values)

# Create an empty white image with a resolution of res
image = Image.new("RGB", (res, res), "white")
draw = ImageDraw.Draw(image)
pixels = image.load()

# Load an image used for the celestial sphere background
background = Image.open("../background/patterned_circles.png")
px_bg = background.load()

counter = 0
# Loop through each (β, γ) coordinate on the observer's plane
for i in range(len(beta)):
    for j in range(len(gamma)):
        # Solve the geodesic equations for the current (β, γ) with given initial conditions
        init_p = obs.p_init(gamma[i, j], beta[i, j])
        geo = geodesics.Geodesics(init_q, init_p, [0.99])

        ivp = np.zeros(shape=8, dtype=float)
        ivp[1:3] = init_q
        ivp[4:] = init_p

        sol = kdp45(func=geo.hamilton_eqs,
                    init=ivp,
                    t_init=0.,
                    h_init=5.,
                    num_iter=10_000)
        print(sol)      # може да има проблеми с нулеви начални стойности, щото гърми
        # Temporary print statement for debugging
        print(i, j)

        # това надолу не е вярно, но ме мързи сега да го оправям
        # Check if the light ray falls into the black hole; if not, map it to the celestial sphere
        for k in range(len(sol.y[1])):
            # TODO: да направя оценка на Δr
            # TODO: да поправя условието за падане в дупката
            if abs(sol.y[1, k]) <= r_plus + 1e-1:   # <- много грешно
                # The light ray falls into the black hole; set the pixel to black
                pixels[i, j] = (0, 0, 0)
                print("yeeeeet")
                break
            # временно, в събота имаше проблеми с интегрирането
            try:
                if abs(abs(sol.y[1, k]) - 30) < 1e-1:
                    # Light ray hits the celestial sphere; map it to a pixel on the background image
                    coord = np.asarray([sol.y[2, k], sol.y[3, k]]) / np.pi
                    print(coord)
                    px_coord = map_angle_to_pixel(coord[0], coord[1], res)
                    print(px_coord)
                    pixels[i, j] = px_bg[px_coord[0], px_coord[1]]
                    # TODO: да свържа пикселите с импакт параметрите
                    break
            except IndexError:
                pixels[i, j] = (0, 0, 0)
                counter += 1


# в момента е грешно
image.save("test_hole.png")

# TODO: да начертая графиката
fig, ax = plt.subplots()
fig.suptitle(r'Kerr Black Hole Shadow for $\alpha=0.99$')
#ax.imshow(pixels)
# plt.imshow(image_data в пиксели, cmap='gray', interpolation='nearest')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
#ax.legend(facecolor="white", edgecolor="white", loc='upper right')
#plt.show()
# това е да видя каква част от пикселите са минали условието за падане
print(f"num raised index errors: {counter}")
