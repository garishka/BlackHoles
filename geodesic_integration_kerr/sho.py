from integrator import symplectic_integrator
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def H1D(qp, *params):
    m, omega = params
    q, p = qp
    return p**2/(2*m) + m*omega**2*q**2/2

qp = [0.5, 3.]
res = symplectic_integrator(H1D,
                            qp0=qp,
                            metric_params=[1., .8],
                            step_size=0.1,
                            omega=0.8,
                            num_steps=100,
                            ord=4)

t = np.linspace(0, 10, 100)

plt.style.use('seaborn-v0_8')
fig = plt.figure(layout="constrained")
gs = GridSpec(4, 3, figure=fig)
ax = fig.add_subplot(gs[0:3, :])
ax_qp = fig.add_subplot(gs[3, :])

ax.set_xlabel(r'$q$')
ax.set_ylabel(r'$p$')
#ax.set_aspect('equal')

ax_qp.set_xlabel(r'$t$')
ax.set_aspect('equal')

ax.scatter(qp[0], qp[1], color='green', marker='o')
ax.text(qp[0]+0.1, qp[1]+0.1, "(q0, p0)")
ax.plot(res[0], res[1], linewidth=0.5, color="red")
ax_qp.plot(t, res[0], linewidth=0.5, color="red", label="q")
ax_qp.plot(t, res[1], linewidth=0.5, color="blue", label="p")
ax_qp.legend(loc="best")
plt.savefig("sho_test.pdf")
