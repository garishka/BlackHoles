import numpy as np
from PIL import Image, ImageDraw

import kerrgeodesics
import metricKerr
from geodesic_integration_kerr.kdp_integrator import kdp45
from angle_utils import map_angle_to_pixel, map_celestial
from observer import Observer

# Image resolution
res = 50

# Generate a grid of angle values (δ, γ) ≡ (β, α) for the observer's viewpoint
delta = np.linspace(0.9955*np.pi, 1.0045*np.pi, res)
gamma = np.linspace(-0.0045*np.pi, 0.0045*np.pi, res)

# Define a Kerr black hole with spin parameter α = 0.99
black_hole = metricKerr.KerrBlackHole(alpha=0.99)
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
        geo = kerrgeodesics.KerrGeodesics(init_q, init_p, [0.99])

        ivp = np.zeros(shape=8, dtype=float)
        ivp[1:3] = init_q
        ivp[4:] = init_p

        tau, z = kdp45(func=geo.hamilton_eqs,
                       init=ivp,
                       t_init=1e-10,
                       h_init=.9,  # няма значение каква е стъпката, приближава се много бавно, не знам дали ще доживея
                       num_iter=70_000)
        # Temporary print statement for debugging
        print(i, j)
        print(tau)
        print(len(z[0]))
        print(z)

        # това надолу не е вярно, но ме мързи сега да го оправям
        # Check if the light ray falls into the black hole; if not, map it to the celestial sphere
        for k in range(len(z[1])):
            # TODO: да направя оценка на Δr
            if z[1, k] <= r_plus + 1e-1:
                # The light ray falls into the black hole; set the pixel to black
                pixels[i, j] = (0, 0, 0)
                print("yeeeeet")

                # временно, в събота имаше проблеми с интегрирането
                try:
                    if abs(z[1, k] - obs.coord()[0]) < 1e-1:
                        # Light ray hits the celestial sphere; map it to a pixel on the background image
                        coord = np.asarray([z[2, k] % (2 * np.pi), z[3, k] % np.pi])
                        print(coord)
                        px_coord = map_angle_to_pixel(coord[0], coord[1], res,
                                                  [[-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2]])
                        print(px_coord)
                        pixels[i, j] = px_bg[px_coord[0], px_coord[1]]
                        # TODO: да свържа пикселите с импакт параметрите
                        break
                except IndexError:
                    pixels[i, j] = (0, 0, 0)
                    counter += 1

                break


image.save("test_hole.png")

# това е да видя каква част от пикселите са минали условието за падане
print(f"num raised index errors: {counter}")
