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
RES = 500

# Generate a grid of angle values (δ, γ) ≡ (β, α) for the observer's viewpoint
delta = np.linspace(0.9955 * np.pi, 1.0045 * np.pi, RES)
gamma = np.linspace(-0.0045 * np.pi, 0.0045 * np.pi, RES)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = kerrgeodesics.KerrBlackHole()
r_plus = black_hole.r_plus()

# Set the observer's position to (r, θ, ϕ) = (500, π, 0), a.k.a default position
obs = Observer()
init_q = obs.coord()

# defining the values of the four corners of the image
# because of the symmetry of the interval, 2 points are enough (or even 1)
# those are the coordinates of the non-flipped image
ll_corner = obs.impact_params(gamma[0], delta[0])       # [[ 7.06859738] [-7.06859738]]
lr_corner = obs.impact_params(gamma[0], delta[-1])
# ul_corner = np.array([ll_corner[0], -ll_corner[1]])
# ur_corner = np.array([lr_corner[0], -lr_corner[-1]])
avg_x = (ll_corner[0] + lr_corner[0]) / 2       # [-1570.79941818] center around zero
ll_corner[0] -= avg_x
lr_corner[0] -= avg_x
print(ll_corner)



# Load an image used for the celestial sphere background
# background = Image.open("../background/patterned_circles.png")
# px_bg = background.load()


def solve_BH_shadow(delta_i, gamma_j):
    logging.info(f"Processing pixel ({delta_i}, {gamma_j})")

    # Solve the geodesic equations for the current (δ, γ) ≡ (β, α) ≡ (-x/r, y/r) with given initial conditions
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

    image.save("test_bh_alpha0p99.png")
