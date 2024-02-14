import itertools
import time
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
import logging

import kerrgeodesics
# from geodesic_integration_kerr.kdp_integrator import kdp45
from geodesic_integration_kerr.dormand_prince import RK45
from observer import Observer

# Image resolution
RES = 50

# Generate a grid of angle values (δ, γ) ≡ (β, α) for the observer's viewpoint
delta = np.linspace(0.9955 * np.pi, 1.0045 * np.pi, RES)
gamma = np.linspace(-0.0045 * np.pi, 0.0045 * np.pi, RES)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = kerrgeodesics.KerrBlackHole()
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (500, π, 0), a.k.a default position
obs = Observer()
init_q = obs.coord()

# Load an image used for the celestial sphere background
background = Image.open("../background/patterned_circles.png")
px_bg = background.load()


def solve_BH_shadow(delta_i, gamma_j):
    logging.info(f"Processing pixel ({delta_i}, {gamma_j})")

    # Solve the geodesic equations for the current (δ, γ) ≡ (β, α) ≡ (-x/r, y/r) with given initial conditions
    init_p = obs.p_init(delta[delta_i], gamma[gamma_j])
    geo = kerrgeodesics.KerrGeodesics()

    ivp = np.zeros(shape=8, dtype=float)
    ivp[1:3] = init_q
    ivp[4:] = init_p

    fallen, tau, z = RK45(func=geo.hamilton_eqs,
                          init=ivp,
                          t_interval=(-10_000, 1e-10),
                          h_init=.9,
                          r_plus=r_plus)

    return fallen


if __name__ == "__main__":

    # Create an empty white image with a resolution of RES
    image = Image.new("RGB", (RES, RES), "white")
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    logging.basicConfig(level=logging.INFO)
    iterable = list(itertools.product(range(RES), range(RES)))
    i, j = zip(*iterable)
    
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Loop through each (δ, γ) coordinate on the observer's plane
        for fallen in executor.map(solve_BH_shadow, i, j):

            if fallen:
                # The light ray falls into the black hole; set the pixel to black
                pixels[i, j] = (0, 0, 0)

    end = time.time()
    elapsed_time_sec = end - start
    hours, remainder = divmod(elapsed_time_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {int(hours)}:{int(minutes)}:{seconds}")

    image.save("test_hole.png")
