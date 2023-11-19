# TODO: set up на уравненията на Хамилтън във форма, удобна за метода
# TODO: да напиша интегратора
import numpy as np
import warnings
from dual import DualNumber


class SymplecticIntegrator:
    def __init__(self, q0, p0, metric_params, steps, order, null=True):
        self.q0 = q0
        self.p0 = p0
        self.metric_params = metric_params
        self.steps = steps
        self.null = null
        self.order = order


