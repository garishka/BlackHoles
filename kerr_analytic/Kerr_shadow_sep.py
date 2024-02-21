# c = G = 1
# В момента този код не е направен reusable – пренаписавала съм неща колкото да изкарам каквото ми трябва, без много
# мисъл дали миналите неща ще ми потрябват някъде/някога.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp

rot_params_small = [0.01, 0.05, 0.1]
rot_params = [0.01, 0.5, 0.8, 0.9, 0.999]
colors = ['darkviolet', 'mediumblue', 'mediumspringgreen', 'gold', 'red']

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


# направих частните случаи първо да видя дали се получава нещо читаво
def shadow_rim_small_a(alpha, M=4):
    # theta=pi/2
    # (8.6.34) от Фролов
    phi = np.linspace(0, 2* np.pi, 100)
    x = 3* np.sqrt(3) * M * np.cos(phi) + 2 * alpha * M
    y = 3* np.sqrt(3) * M * np.sin(phi)
    return x, y


def shadow_rim_big_a(M=4):
    # theta=pi/2
    # (8.3.38) от Фролов
    # най-вероятно трябва да променя интервала подобно на общия случай, вместо да добавям на ръка правата x=-2M
    x = np.linspace(-2*M, 7*M, 100)
    y_plus = np.sqrt((M+np.sqrt(M*(2*M+x)))**3 * (3*M-np.sqrt(M*(2*M+x))))/M
    return x, y_plus


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


#fig = plt.figure(figsize=(13, 4))
#gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
#fig.suptitle(r'Kerr Black Hole Shadow Rim for $M=1$')

# за ектремалната черна дупка
#x, y_plus = shadow_rim_big_a()
#ax.plot(x, y_plus, label=f'a = 1', color=colors[4])
#ax.plot(x, -y_plus, color=colors[4])
#ax.plot(0, 0, 'o')
#ax.hlines(0, -10, 30, color='blue', linewidth=0.7)
#ax.vlines(0, -20, 20, color='blue', linewidth=0.7)
#ax.vlines(-8, -6.92820323,  6.92820323, color='red')
#ax.text(0.2, 0.4, "(0, 0)", fontsize=12, color='blue')

th = [0.001, np.pi/4, np.pi/2 ]
th_label = [r'$0.001$ rad',  r'$\pi/4$', r'$\pi/2$']
# for i in range(3):
#     for alpha in rot_params:
#         x, y = shadow_rim(alpha, theta=th[i], M=1)
#         ax = plt.subplot(gs[i])
#         ax.plot(x, y, label=f'a = {alpha}', color=colors[rot_params.index(alpha)])
#         ax.plot(x, -y, color=colors[rot_params.index(alpha)])

#     ax.set_xlabel(r'$x$, [M]')
#     ax.set_ylabel(r'$y$, [M]')
#     ax.set_aspect('equal', adjustable='datalim')
#     ax.legend(facecolor="white", edgecolor="white", loc='upper right', fontsize='8')
#     ax.set_title(f'$\\theta_0=$'+th_label[i])
#     ax.grid()

# plt.tight_layout()
# plt.savefig('kerr_shadowM.pdf')
