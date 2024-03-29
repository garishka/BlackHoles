import itertools
import time
import numpy as np
from typing import Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
from PIL import Image, ImageDraw
import concurrent.futures


from math_tools import dual, blackhole_kdp

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


@dataclass
class MetricFunctions:
    """Non-zero components of a metric, describing an axially symmetric space-time in spherical-like coordinates."""
    g00: Callable
    g11: Callable
    g22: Callable
    g33: Callable
    g03: Callable
    g30: Callable

    def metric(self, state_vec, metric_params):
        g = np.zeros(shape=(4, 4))

        g[0, 0] = self.g00(*state_vec, *metric_params)
        g[1, 1] = self.g11(*state_vec, *metric_params)
        g[2, 2] = self.g22(*state_vec, *metric_params)
        g[3, 3] = self.g33(*state_vec, *metric_params)
        g[0, 3] = g[3, 0] = self.g03(*state_vec, *metric_params)

        return g


def get_g_inverse(cov_g: MetricFunctions) -> MetricFunctions:
    def a(state_vector, params):
        return cov_g.g00(*state_vector, *params) * cov_g.g33(*state_vector, *params) - cov_g.g03(*state_vector, *params) ** 2

    g_inverse = MetricFunctions()
    g_inverse.g00 = lambda state_vector, params: cov_g.g33(*state_vector, *params) / a(state_vector, params)
    g_inverse.g11 = lambda state_vector, params: 1 / cov_g.g11(*state_vector, *params)
    g_inverse.g22 = lambda state_vector, params: 1 / cov_g.g22(*state_vector, *params)
    g_inverse.g33 = lambda state_vector, params: cov_g.g00(*state_vector, *params) / a(state_vector, params)
    g_inverse.g03 = lambda state_vector, params: - cov_g.g03(*state_vector, *params) / a(state_vector, params)

    return g_inverse


class RayTracer:

    def __init__(self,
                 cov_metric: MetricFunctions,
                 observer_state_vector: Union[list, np.ndarray],
                 metric_params: Union[list, np.ndarray],
                 fall_cond: Callable,
                 cond_tolerance: float,
                 alpha_interval: Union[tuple, list, float],
                 beta_interval: Union[tuple, list, float],
                 resolution: int,
                 shadow: bool,
                 trajectory: bool,
                 end_state: bool):
        self.cov_metric = cov_metric
        self.observer_state_vector = observer_state_vector
        self.metric_params = metric_params
        self.fall_cond = fall_cond
        self.cond_tolerance = cond_tolerance
        self.alpha_interval = alpha_interval
        self.beta_interval = beta_interval
        self.resolution = resolution
        self.shadow = shadow
        self.trajectory = trajectory
        self.end_state = end_state

        if self.trajectory and isinstance(self.alpha_interval, list):
            alpha_values = np.array(self.alpha_interval)
            beta_values = np.array(self.beta_interval)
        else:
            alpha_values = np.linspace(*self.alpha_interval, self.resolution)
            beta_values = np.linspace(*self.beta_interval, self.resolution)

    def initial_conditions(self, alpha, beta):
        """Initial conditions for the light rays in ZAMO frame."""
        g = self.cov_metric.metric(*self.observer_state_vector, *self.metric_params)

        gamma = -g[0, 3] / np.sqrt(g[3, 3] * (g[0, 3] ** 2 - g[0, 0] * g[3, 3]))
        zeta = np.sqrt(g[3, 3] / (g[0, 3] ** 2 - g[0, 0] * g[3, 3]))

        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)
        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)

        qp0 = np.zeros(shape=(8,), dtype=float)

        qp0[1] = self.observer_state_vector[0]
        qp0[2] = self.observer_state_vector[1]

        qp0[4] = (1 + gamma * np.sqrt(g[3, 3]) * sin_b * cos_a) / zeta
        qp0[5] = np.sqrt(g[1, 1]) * cos_a * cos_b
        qp0[6] = np.sqrt(g[2, 2]) * sin_a
        qp0[7] = np.sqrt(g[3, 3]) * sin_b * cos_a

        return qp0

    def impact_parameters(self, alpha, beta):
        r_perimetral = np.sqrt(self.cov_metric.metric(*self.observer_state_vector, *self.metric_params)[3, 3])
        return - r_perimetral * beta, r_perimetral * alpha

    def get_edges(self):
        """Returns coordinates of the lower left and upper right corner of the image"""
        xll, yll, xlr, yul = self.impact_parameters(self.alpha_interval[0],self.beta_interval[0]), self.impact_parameters(self.alpha_interval[-1], self.beta_interval[-1])
        return xll, xlr, yll, yul

    def hamiltons_equations(self, l: float, z: np.ndarray):
        metric = get_g_inverse(self.cov_metric)
        g = metric.metric(*z[1:3], *self.metric_params)

        dg00dr = dual.partial_deriv(metric.g00, z[1:3], 0, self.metric_params)
        dg11dr = dual.partial_deriv(metric.g11, z[1:3], 0, self.metric_params)
        dg22dr = dual.partial_deriv(metric.g22, z[1:3], 0, self.metric_params)
        dg33dr = dual.partial_deriv(metric.g33, z[1:3], 0, self.metric_params)
        dg03dr = dual.partial_deriv(metric.g03, z[1:3], 0, self.metric_params)

        dg00dth = dual.partial_deriv(metric.g00, z[1:3], 1, self.metric_params)
        dg11dth = dual.partial_deriv(metric.g11, z[1:3], 1, self.metric_params)
        dg22dth = dual.partial_deriv(metric.g22, z[1:3], 1, self.metric_params)
        dg33dth = dual.partial_deriv(metric.g33, z[1:3], 1, self.metric_params)
        dg03dth = dual.partial_deriv(metric.g03, z[1:3], 1, self.metric_params)

        dzdl = np.zeros(shape=8, dtype=float)

        dzdl[0] = g[0, 0] * z[4] + g[0, 3] * z[7]
        dzdl[1] = g[1, 1] * z[5]
        dzdl[2] = g[2, 2] * z[6]
        dzdl[3] = g[0, 3] * z[4] + g[3, 3] * z[7]

        dzdl[4] = 1e-15
        dzdl[5] = - 0.5 * (dg00dr * z[4] ** 2 + 2 * dg03dr * z[4] * z[7] + dg11dr * z[5] ** 2 + dg22dr * z[6] ** 2 + dg33dr * z[7] ** 2)
        dzdl[6] = - 0.5 * (dg00dth * z[4] ** 2 + 2 * dg03dth * z[4] * z[7] + dg11th * z[5] ** 2 + dg22dth * z[6] ** 2 + dg33dth * z[7] ** 2)
        dzdl[7] = 1e-15

        return dzdl

    def solve_shadow(self, beta_i, alpha_j):
        fallen, trajectory = blackhole_kdp.RK45_mod(func=self.hamiltons_equations,
                                                    init=self.initial_conditions(alpha_j, beta_i),
                                                    t_interval=(-1e5, 1e-10),
                                                    r_plus=self.fall_cond(*self.observer_state_vector, *self.metric_params),
                                                    delta_r=self.cond_tolerance,
                                                    trajectory=True)

        if self.shadow:
            return fallen, beta_i, alpha_j
        if self.end_state:
            return trajectory.transpose()[-1], beta_i, alpha_j
        if self.trajectory:
            return trajectory, beta_i, alpha_j

    def draw_shadow(self, title: str, name_file: str):
        image = Image.new("RGB", (self.resolution, self.resolution), "white")
        draw = ImageDraw.Draw(image)
        pixels = image.load()

        iterable = list(itertools.product(range(self.resolution), range(self.resolution)))
        i, j = zip(*iterable)

        start = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Loop through each (δ, γ) coordinate on the observer's plane
            for fallen, delta_i, gamma_j in executor.map(self.solve_shadow, i, j):

                if fallen:
                    # The light ray falls into the black hole; set the pixel to black
                    pixels[delta_i, gamma_j] = (0, 0, 0)

        end = time.time()
        elapsed_time_sec = end - start
        hours, remainder = divmod(elapsed_time_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time taken: {int(hours)}:{int(minutes)}:{seconds}")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image, extent=[*self.get_edges()])  # да напиша после границите
        ax.set_xlabel(r'$x$, [M]')
        ax.set_ylabel(r'$y$, [M]')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(facecolor="white", edgecolor="white", loc='upper right')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(name_file)


    def draw_trajectory(geo_solution: np.ndarray, title: str, name_file: str):
        ax = plt.figure().add_subplot(projection='3d')
        # да добавя как ще изглежда проекцията за наблюдател
        ax.plot(geo_solution[1:3])      # това е грешно, просто го слагам за пълнеж
        ax.legend(facecolor="white", edgecolor="white", loc='upper right')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(name_file)

