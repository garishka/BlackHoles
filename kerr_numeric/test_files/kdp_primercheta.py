from math_tools.kdp_integrator import kdp45

from matplotlib import pyplot as plt
import numpy as np
import inspect
import time


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


# може да се използва за проверка колко параметъра се пускат на функцията за интегриране
print(len(inspect.signature(periodic_uwu).parameters))
# -> 3
print(inspect.signature(kdp45))
# -> (func: Callable, init: Union[numpy.ndarray, List], t_init: float, h_init: float, num_iter: int, **params)

start = time.time()
t_per, y_per = kdp45(func=periodic_uwu,
                     init=[.8],
                     t_init=0.,
                     h_init=0.01,
                     num_iter=200,
                     params=5)
end = time.time()

t = np.linspace(t_per[0], t_per[-1], 2000)
anl_per = - np.cos(5 * t) + 1.8

print(end-start)
# normal: 0.011017084121704102s


def polynomial(t, y, d, e):
    return d * t ** 3 + e / 2 * t ** 2 - t


t_poly, y_poly = kdp45(func=polynomial,
                       init=[.5],
                       t_init=0.,
                       h_init=0.001,
                       num_iter=500,
                       e=2.3,
                       d=2.4,       # останалите параметри ще бъдат махнати
                       g=9.8,
                       h=6.63)


anl_poly = 2.3 / 4 * t_poly ** 4 + 2.3 / 6 * t_poly ** 3 - 0.5 * t_poly ** 2 - 159.733333333

fig, ax = plt.subplots()
ax.scatter(t_per, y_per[0], color="red")
ax.plot(t, anl_per, color="black")
#ax.scatter(t_exp, y_exp[1], color="blue")
plt.show()
