from geodesic_integration_kerr.kdp_integrator import kdp45
from matplotlib import pyplot as plt
import numpy as np


def exponential_decay(t, y, d):
    return - d * y


t_exp, y_exp = kdp45(func=exponential_decay,
                     init=[10.],
                     t_init=0,
                     h_init=0.1,
                     num_iter=100,
                     params=5)

anl_exp = 10 * np.exp(-5 * t_exp)


def periodic_uwu(t, y, d):
    return d * np.sin(d * t)


t_per, y_per = kdp45(func=periodic_uwu,
                     init=[.8],
                     t_init=0,
                     h_init=0.1,
                     num_iter=100,
                     params=5)

t = np.linspace(t_per[0], t_per[-1], 200)
anl_per = - np.cos(5 * t) + 1.8


def polynomial(t, y, d):
    return d * t ** 3 + d / 2 * t ** 2 - t


t_poly, y_poly = kdp45(func=polynomial,
                       init=[4.5],
                       t_init=4.,
                       h_init=0.1,
                       num_iter=500,
                       params=2.3)


anl_poly = 2.3 / 4 * t_poly ** 4 + 2.3 / 6 * t_poly ** 3 - 0.5 * t_poly ** 2 - 159.733333333
print(anl_poly)

fig, ax = plt.subplots()
ax.scatter(t_poly, y_poly[0], color="red")
ax.plot(t_poly, anl_poly, color="black")
#ax.scatter(t_exp, y_exp[1], color="blue")
plt.show()
