import itertools
import time
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
import logging

import geodesics
from math_tools.blackhole_kdp import RK45_mod
from observer import Observer

# Image resolution
RES = 100

# Deformation parameter
g = np.inf
# Declination angle
th = np.pi / 2

# Generate a grid of angle values (β, α) for the observer's viewpoint
beta = np.linspace(0.9955 * np.pi, 1.0045 * np.pi, RES)
alpha = np.linspace(-0.0045 * np.pi, 0.0045 * np.pi, RES)


# Define a Gamma mimicker with γ = 0.51
black_hole = geodesics.GammaMimicker(gamma=g)
r_sing = black_hole.r_inf_redshift()

# Set the observer's position to (r, θ, ϕ) = (500, π, 0), a.k.a default position
obs = Observer(position=np.array([500, th]), gamma=g)


def solve_BH_shadow(delta_i, gamma_j):
    # Solve the geodesic equations for the current (δ, γ) ≡ (β, α) ≡ (-x/r, y/r) with given initial conditions
    logging.info(f"Processing pixel ({delta_i}, {gamma_j})")

    ivp = obs.qp_zamo(alpha[gamma_j], beta[delta_i])
    geo = geodesics.GammaInftyGeodesics()

    fallen, end_values = RK45_mod(func=geo.hamilton_eqs,
                                  init=ivp,
                                  t_interval=(-1e5, 1e-10),
                                  h_init=9.,
                                  r_plus=r_sing,
                                  delta_r=1e-2,
                                  trajectory=False)

    return fallen, end_values, delta_i, gamma_j


if __name__ == "__main__":

    # Create an empty white image with a resolution of RES
    image = Image.new("RGB", (RES, RES), "white")
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    logging.basicConfig(level=logging.INFO)
    iterable = list(itertools.product(range(RES), range(RES)))
    i, j = zip(*iterable)

    #results = np.empty(shape=(RES, RES, 8))

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Loop through each (δ, γ) coordinate on the observer's plane
        for fallen, end_vals, delta_i, gamma_j in executor.map(solve_BH_shadow, i, j):

            if fallen:
                # The light ray falls into the black hole; set the pixel to black
                pixels[delta_i, gamma_j] = (0, 0, 0)
            #else:
            #    results[delta_i, gamma_j] = end_vals

    end = time.time()
    elapsed_time_sec = end - start
    hours, remainder = divmod(elapsed_time_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {int(hours)}:{int(minutes)}:{seconds}")

    #np.save("gamma_1p0_results.npy", results)
    # To load array
    # data = np.load('data.npy')

    image.save("gamma_infty_shadow.png")
