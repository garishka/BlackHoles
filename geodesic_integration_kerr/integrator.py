import numpy as np
import dual

# molei tao, doi: 10.1103/PhysRevE.94.043303
def flow_A(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    qy = np.concatenate((double_copy[:4], double_copy[12:]), axis=0)

    double_qp[4:8] -= step * np.array([dual.partial_deriv(H, qy, q, *metric_params) for q in qy[:4]])
    double_qp[8:12] += step * np.array([dual.partial_deriv(H, qy, y, *metric_params) for y in qy[4:]])

    return double_qp


def flow_B(H, double_qp, metric_params, step):
    double_copy = double_qp.copy()
    xp = np.concatenate((double_copy[8:12], double_copy[4:8]))

    double_qp[:4] += step * np.array([dual.partial_deriv(H, xp, x, *metric_params) for x in xp[:4]])
    double_qp[8:12] -= step * np.array([dual.partial_deriv(H, xp, p, *metric_params) for p in xp[4:]])

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


def symplectic_integrator(H, qp0, metric_params, step, omega):
    double_qp = np.tile(qp0, 2)

    flow_ar_qp = flow_A(H, double_qp, metric_params, step/2)
    flow_br_qp = flow_B(H, flow_ar_qp, metric_params, step/2)
    flow_c_qp = flow_C(flow_br_qp, step, omega)
    flow_bl_qp = flow_B(H, flow_c_qp, metric_params, step/2)
    double_qp = flow_A(H, flow_bl_qp, metric_params, step/2)

    return double_qp