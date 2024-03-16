import numpy as np
from typing import Union, Callable
from dataclasses import dataclass

from math_tools import dual


@dataclass
class MetricFunctions:
    """Non-zero components of a metric, describing an axially symmetric space-time in spherical-like coordinates."""
    state_vector: Union[list, np.ndarray]
    params: Union[list, np.ndarray]
    g00: Callable
    g11: Callable
    g22: Callable
    g33: Callable
    g03: Callable
    g30: Callable

    def metric(self):
        g = np.zeros(shape=(4, 4))

        g[0, 0] = self.g00(*self.state_vector, *self.params)
        g[1, 1] = self.g11(*self.state_vector, *self.params)
        g[2, 2] = self.g22(*self.state_vector, *self.params)
        g[3, 3] = self.g33(*self.state_vector, *self.params)
        g[0, 3] = g[3, 0] = self.g03(*self.state_vector, *self.params)

        return g


def get_g_inverse(cov_g: MetricFunctions) -> MetricFunctions:
    def a(state_vector, params):
        return cov_g.g00(*state_vector, *params) * cov_g.g33(*state_vector, *params) - cov_g.g03(*state_vector, *params) ** 2

    g_inverse = MetricFunctions()
    g_inverse.state_vector = cov_g.state_vector
    g_inverse.params = cov_g.params
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
                 fall_cond: float,
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
            alpha = np.array(self.alpha_interval)
            beta = np.array(self.beta_interval)
        else:
            alpha = np.linspace(*self.alpha_interval, self.resolution)
            beta = np.linspace(*self.beta_interval, self.resolution)

        g_cov = self.cov_metric.metric()
        g_contra = get_g_inverse(self.cov_metric).metric()

    def initial_conditions(self, alpha, beta):
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



