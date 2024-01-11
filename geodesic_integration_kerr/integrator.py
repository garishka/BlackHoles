import numpy as np
from geodesic_integration_kerr import dual


# Molei Tao, doi: 10.1103/PhysRevE.94.043303
def _flow_A(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    dim = len(double_qp)
    if dim % 4 != 0:
        raise ValueError("Wrong dimensionality of initial conditions!")
    qy = np.concatenate((double_copy[:int(0.25*dim)], double_copy[int(0.75*dim):]), axis=0)

    for i in range(int(0.25*dim)):
        double_qp[int(0.25*dim)+i] -= step * dual.partial_deriv(H, qy, i, *metric_params)
        double_qp[int(0.5*dim)+i] += step * dual.partial_deriv(H, qy, i+int(0.25*dim), *metric_params)

    return double_qp


def _flow_B(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    dim = len(double_qp)
    if dim % 4 != 0:
        raise ValueError("Wrong dimensionality of initial conditions!")
    xp = np.concatenate((double_copy[int(0.5*dim):int(0.75*dim)], double_copy[int(0.25*dim):int(0.5*dim)]))

    for i in range(int(0.25*dim)):
        double_qp[i] += step * dual.partial_deriv(H, xp, i+int(0.25*dim), *metric_params)
        double_qp[int(0.75*dim)+i] -= step * dual.partial_deriv(H, xp, i, *metric_params)

    return double_qp


def _flow_C(double_qp, step, omega):
    double_copy = double_qp.copy()
    dim = len(double_qp)
    if dim % 4 != 0:
        raise ValueError("Wrong dimensionality of initial conditions!")
    qp = double_copy[:int(0.5*dim)]
    xy = double_copy[int(0.5*dim):]

    I = np.identity(int(0.25*dim), dtype=float)
    ul = lr = np.cos(2 * omega * step) * I
    ur = np.sin(2 * omega * step) * I
    ll = - ur
    R = np.block([[ul, ur], [ll, lr]])

    double_qp[:int(0.5*dim)] = 0.5 * (qp + xy) + 0.5 * R @ (qp - xy)
    double_qp[int(0.5*dim):] = 0.5 * (qp + xy) - 0.5 * R @ (qp - xy)

    return double_qp


def second_order(H, double_qp, metric_params, step_size, omega):

    flow_ar_qp = _flow_A(H, double_qp, metric_params, step_size / 2)
    flow_br_qp = _flow_B(H, flow_ar_qp, metric_params, step_size / 2)
    flow_c_qp = _flow_C(flow_br_qp, step_size, omega)
    flow_bl_qp = _flow_B(H, flow_c_qp, metric_params, step_size / 2)
    double_qp = _flow_A(H, flow_bl_qp, metric_params, step_size / 2)

    return double_qp


def forth_order(H, double_qp, metric_params, step_size, omega):
    gamma = 1./(2 - 2 ** 0.2)

    flow_r = second_order(H, double_qp, metric_params, (step_size * gamma), omega)
    flow_m = second_order(H, flow_r, metric_params, (step_size * (1. - 2. * gamma)), omega)
    flow_l = second_order(H, flow_m, metric_params, (step_size * gamma), omega)

    return flow_l


def symplectic_integrator(H, qp0, metric_params, step_size: float, omega: float, num_steps: int, ord: float):
    double_qp = np.tile(qp0, 2)
    results = np.zeros(shape=(num_steps, 2 * len(qp0)))
    results[0] = double_qp

    if ord == 2:
        for i in range(1, num_steps):
            results[i] = second_order(H, results[i-1], metric_params, step_size, omega)
    elif ord == 4:
        for i in range(1, num_steps):
            results[i] = forth_order(H, results[i-1], metric_params, step_size, omega)

    results_T = np.transpose(results)

    return results_T
