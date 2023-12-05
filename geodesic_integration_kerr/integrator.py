import numpy as np
from geodesic_integration_kerr import dual


# molei tao, doi: 10.1103/PhysRevE.94.043303
def _flow_A(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    qy = np.concatenate((double_copy[:4], double_copy[12:]), axis=0)

    for i in range(4):
        double_qp[4+i] -= step * dual.partial_deriv(H, qy, i, *metric_params)
        double_qp[8+i] += step * dual.partial_deriv(H, qy, i+4, *metric_params)

    return double_qp


def _flow_B(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    xp = np.concatenate((double_copy[8:12], double_copy[4:8]))

    for i in range(4):
        double_qp[i] += step * dual.partial_deriv(H, xp, i+4, *metric_params)
        double_qp[12+i] -= step * dual.partial_deriv(H, xp, i, *metric_params)

    return double_qp


def _flow_C(double_qp, step, omega):
    double_copy = double_qp.copy()
    qp = double_copy[:8]
    xy = double_copy[8:]

    I = np.identity(4, dtype=float)
    ul = lr = np.cos(2 * omega * step) * I
    ur = np.sin(2 * omega * step) * I
    ll = - ur
    R = np.block([[ul, ur], [ll, lr]])

    double_qp[:8] = 0.5 * (qp + xy) + 0.5 * R @ (qp - xy)
    double_qp[8:] = 0.5 * (qp + xy) - 0.5 * R @ (qp - xy)

    return double_qp


def second_order(H, double_qp, metric_params, step_size, omega):

    flow_ar_qp = _flow_A(H, double_qp, metric_params, step_size / 2)
    flow_br_qp = _flow_B(H, flow_ar_qp, metric_params, step_size / 2)
    flow_c_qp = _flow_C(flow_br_qp, step_size, omega)
    flow_bl_qp = _flow_B(H, flow_c_qp, metric_params, step_size / 2)
    double_qp = _flow_A(H, flow_bl_qp, metric_params, step_size / 2)

    return double_qp


def forth_order(H, double_qp, metric_params, step_size, omega):
    gamma = 1/(2 - 2 ** 0.2)

    flow_r = second_order(H, double_qp, metric_params, (step_size * gamma), omega)
    flow_m = second_order(H, flow_r, metric_params, (step_size * (1 - 2 * gamma)), omega)
    flow_l = second_order(H, flow_m, metric_params, (step_size * gamma), omega)

    return flow_l


def symplectic_integrator(H, qp0, metric_params, step_size: float, omega: float, num_steps: int, ord: float):
    double_qp = np.tile(qp0, 2)
    results = np.zeros(shape=(num_steps, 2 * len(qp0)))
    results[0] = double_qp

    if ord == 2:
        for i in range(1, num_steps):
            current_qp = second_order(H, results[i-1], metric_params, step_size, omega)
            results[i] = current_qp
    elif ord == 4:
        for i in range(1, num_steps):
            current_qp = forth_order(H, results[i-1], metric_params, step_size, omega)
            results[i] = current_qp

    results_T = np.transpose(results)

    return results_T
