import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sympy as sp
import matplotlib as mpl

from plot import ll_corner

# за графиките
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


def l_expr(r, M, a):
    return - (r ** 3 - 3 * M * r ** 2 + a ** 2 * (r + M))/(a * (r - M))


def Q_expr(r, M, a):
    return r ** 3 * (4 * M * a ** 2 - r * (r - 3 * M) ** 2) / (a * (r - M)) ** 2


def shadow_rim(alpha, theta=np.pi/2, M=1):
    a = alpha * M

    # определяне на границите на изменение на r от Q>=0
    r = sp.symbols('r')
    l = l_expr(r, M, a)
    Q = Q_expr(r, M, a)
    poly = Q - (l/np.tan(theta)) ** 2 + (a*np.cos(theta)) ** 2
    roots = sp.solve(poly, r)
    roots = [root.evalf() for root in roots if root.is_real]
    roots = [root.evalf() for root in roots if root >= M]
    # крайните точки са отместени защото водят до имагинерни корени
    # стойностите, с които са отместени, са определени на око, така че да има по-малко прекъсване в кривите на графиката
    r_n = np.linspace(float(roots[0])+1e-7, float(roots[1])-1e-7, 800)

    # пресмятане на прижелните параметри по (8.6.13) + (8.6.20) от Фролов
    x = - l_expr(r_n, M, a) / np.sin(theta)
    y = np.sqrt(Q_expr(r_n, M, a) - (x ** 2 - a ** 2) * (np.cos(theta)) ** 2)

    return x, y


bh_analytic = shadow_rim(alpha=0.99, theta=np.pi/2, M=1)


bh_numerical = Image.open("/kerr_numeric/numeric_kerr_rk45/test_bh_alpha0p99.png")
bh_numerical = bh_numerical.transpose(method=Image.FLIP_LEFT_RIGHT)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(bh_numerical, extent=[ll_corner[-1], -ll_corner[-1], ll_corner[-1], -ll_corner[-1]])
ax.plot(bh_analytic[0], bh_analytic[-1], color="red", label=r"analytic")
ax.plot(bh_analytic[0], -bh_analytic[-1], color="red")
ax.set_xlabel(r'$x$, [M]')
ax.set_ylabel(r'$y$, [M]')
ax.set_aspect('equal', adjustable='datalim')
ax.legend(facecolor="white", edgecolor="white", loc='upper right')
ax.set_title(r"$\alpha=0.99$")
plt.tight_layout()
plt.savefig('check_solution_alpha0p99.pdf')
