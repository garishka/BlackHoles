import itertools
import time
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
import logging

import kerrgeodesics
from math_tools.blackhole_kdp import RK45_mod
from observer import Observer

# Image resolution
RES = 100

# Generate a grid of angle values (δ, γ) ≡ (β, α) for the observer's viewpoint
delta = np.linspace(0.9955 * np.pi, 1.0045 * np.pi, RES)
gamma = np.linspace(-0.0045 * np.pi, 0.0045 * np.pi, RES)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = kerrgeodesics.KerrBlackHole(alpha=0.01)
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (500, π, 0), a.k.a default position
obs = Observer(alpha=0.01)
init_q = obs.coord()

# Due to the symmetry of the interval, defining just one point adequately characterizes both the x and y impact parameter intervals.
# Note that 'll_corner' refers to the lower left corner in the angle space.
# However, in the impact parameter space, this point transforms to the lower right corner.
# Utilizing 'll_corner[-1]' is preferable here as 'll_corner[0]' involves an offset and is less convenient to work with.
ll_corner = obs.impact_params(gamma[0], delta[0])


def solve_BH_shadow(delta_i, gamma_j):
    # Solve the geodesic equations for the current (δ, γ) ≡ (β, α) ≡ (-x/r, y/r) with given initial conditions
    logging.info(f"Processing pixel ({delta_i}, {gamma_j})")

    init_p = obs.p_init(delta[delta_i], gamma[gamma_j])
    geo = kerrgeodesics.KerrGeodesics()

    ivp = np.zeros(shape=8, dtype=float)
    ivp[1:3] = init_q
    ivp[4:] = init_p

    fallen, end_values = RK45_mod(func=geo.hamilton_eqs,
                                  init=ivp,
                                  t_interval=(-10_000, 1e-10),
                                  h_init=.9,
                                  r_plus=r_plus,
                                  trajectory=False)

    return fallen, delta_i, gamma_j


if __name__ == "__main__":

    # Create an empty white image with a resolution of RES
    image = Image.new("RGB", (RES, RES), "white")
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    logging.basicConfig(level=logging.INFO)
    iterable = list(itertools.product(range(RES), range(RES)))
    i, j = zip(*iterable)

    results = np.empty(shape=(RES, RES, 8))

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Loop through each (δ, γ) coordinate on the observer's plane
        for fallen, delta_i, gamma_j in executor.map(solve_BH_shadow, i, j):

            if fallen:
                # The light ray falls into the black hole; set the pixel to black
                pixels[delta_i, gamma_j] = (0, 0, 0)

    end = time.time()
    elapsed_time_sec = end - start
    hours, remainder = divmod(elapsed_time_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {int(hours)}:{int(minutes)}:{seconds}")

    np.save("BH_shadow_results.npy", results)
    # To load array
    # data = np.load('data.npy')

    image.save("test_bh_alpha0p01.png")
