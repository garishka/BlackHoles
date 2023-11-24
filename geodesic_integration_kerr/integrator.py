import numpy as np
from geodesic_integration_kerr import dual


# molei tao, doi: 10.1103/PhysRevE.94.043303
def flow_A(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    qy = np.concatenate((double_copy[:4], double_copy[12:]), axis=0)

    for i in range(4):
        double_qp[4:8] -= step * dual.partial_deriv(H, qy, i, *metric_params)
        double_qp[8:12] += step * dual.partial_deriv(H, qy, i+4, *metric_params)

    return double_qp


def flow_B(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    xp = np.concatenate((double_copy[8:12], double_copy[4:8]))

    for i in range(4):
        double_qp[:4] += step * dual.partial_deriv(H, xp, i, *metric_params)
        double_qp[8:12] -= step * dual.partial_deriv(H, xp, i+4, *metric_params)

    return double_qp


def flow_C(double_qp, step, omega):
    double_copy = double_qp.copy()
    qp = double_copy[:8]
    xy = double_copy[8:]

    I = np.identity(4, dtype=float)
    ul = lr = np.cos(2 * omega * step) * I
    ur = np.sin(2 * omega * step) * I
    ll = - ur
    R = np.block([[ul, ur], [ll, lr]])

    double_qp[:8] = 0.5 * (qp - xy) + R @ (qp - xy)
    double_qp[8:] = 0.5 * (qp + xy) - R @ (qp - xy)

    return double_qp


def second_order(H, double_qp, metric_params, step_size, omega):

    flow_ar_qp = flow_A(H, double_qp, metric_params, step_size / 2)
    flow_br_qp = flow_B(H, flow_ar_qp, metric_params, step_size / 2)
    flow_c_qp = flow_C(flow_br_qp, step_size, omega)
    flow_bl_qp = flow_B(H, flow_c_qp, metric_params, step_size / 2)
    double_qp = flow_A(H, flow_bl_qp, metric_params, step_size / 2)

    return double_qp


def symplectic_integrator(H, qp0, metric_params, step_size, omega, num_steps):
    double_qp = np.tile(qp0, 2)
    results = np.zeros(shape=(num_steps, 2 * len(qp0)))
    results[0] = double_qp

    for i in range(num_steps-1):
        current_qp = second_order(H, results[i], metric_params, step_size, omega)
        results[i+1] = current_qp

    results_T = np.transpose(results)

    return results_T