import numpy as np
from PIL import Image, ImageDraw

import kerrgeodesics
# from geodesic_integration_kerr.kdp_integrator import kdp45
from geodesic_integration_kerr.dormand_prince import RK45
from angle_utils import map_angle_to_pixel, map_celestial
from observer import Observer
import concurrent.futures


# Image resolution
res = 50

# Generate a grid of angle values (δ, γ) ≡ (β, α) for the observer's viewpoint
delta = np.linspace(0.9955*np.pi, 1.0045*np.pi, res)
gamma = np.linspace(-0.0045*np.pi, 0.0045*np.pi, res)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = kerrgeodesics.KerrBlackHole()
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (500, π, 0), a.k.a default position
obs = Observer()
init_q = obs.coord()

# Impact parameters (np.ndarray), calculated at every angle (γ, β)
x, y = obs.impact_params(a1=gamma, a2=delta)

# Create an empty white image with a resolution of res
image = Image.new("RGB", (res, res), "white")
draw = ImageDraw.Draw(image)
pixels = image.load()

# Load an image used for the celestial sphere background
background = Image.open("../background/patterned_circles.png")
px_bg = background.load()

counter = 0
# Loop through each (δ, γ) coordinate on the observer's plane
for i in range(len(delta)):
    for j in range(len(gamma)):
        # Solve the geodesic equations for the current (δ, γ) ≡ (β, α) ≡ (-x/r, y/r) with given initial conditions
        init_p = obs.p_init(delta[i], gamma[j])
        geo = kerrgeodesics.KerrGeodesics(init_q, init_p)

        ivp = np.zeros(shape=8, dtype=float)
        ivp[1:3] = init_q
        ivp[4:] = init_p

        fallen, tau, z = RK45(func=geo.hamilton_eqs,
                              init=ivp,
                              t_interval=(-10_000, 1e-10),
                              h_init=.9,
                              BH=True,
                              r_plus=r_plus)
        # Temporary print statement for debugging
        print(i, j)
        # print(tau)
        print(len(z[1]))

        if fallen:
            # The light ray falls into the black hole; set the pixel to black
            pixels[i, j] = (0, 0, 0)
            print("yeeeeet")


image.save("test_hole.png")
