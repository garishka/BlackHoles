import unittest
from math_tools import symplectic
import numpy as np
from matplotlib import pyplot as plt

# Molei Tao, 2016, "Explicit symplectic approximation of nonseparable Hamiltonians:
# Algorithm and long time performance", DOI: 10.1103/PhysRevE.94.043303


class TestSecondOrder(unittest.TestCase):
    def test_example_one(self):
        def hamiltonian(qp):
            q, p = qp[0], qp[1]
            return 0.5 * (q**2 + 1) * (p**2 + 1)

        def analytic(qp):
            q, p = qp[0], qp[1]

        qp = np.asarray([-3, 1e-8])     # p(0)=0, q(0)<0  според статията
        results = integrator.symplectic_integrator(hamiltonian, qp, [], 1e-1, 20, 1000, 4)

        qp = [results[0], results[3]]
        h = hamiltonian(qp)
        delta_h = h[0] - h
        t = np.linspace(0, 1000, 1000)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(218.62 / 72, 6 * 218.62 / (4 * 72)))
        ax1.plot(results[0], results[3], linewidth=0.5, color="black")
        ax2.plot(t, delta_h, linewidth=0.5, color="black")      # близо 3 пъти по-голяма грешка от тази в статията,
        # грешката при 4ти ред е по-голяма от тази при втори (???????????????????????)
        plt.show()      # изглежда по същия начин като в статията

    def test_example_two(self):
        def hamiltonian(qp):
            q = qp[:3]
            p = qp[3:]
            return 0.5 * (p[0]**2/(1-2/q[1]) - (1-2/q[1])*p[1]**2 - p[2]**2/q[1]**2)

        qp0 = np.array([0, 20, 0, 0.982, 0, -4.472])
        sol = integrator.symplectic_integrator(hamiltonian, qp0, [], 0.2, 2, 50_000, 4)
        t = np.linspace(0, 50_000, 50_000)

        fig, ax1 = plt.subplots(figsize=(218.62 / 72, 6 * 218.62 / (4 * 72)))
        #ax1.plot(t, sol[0], linewidth=0.5, color="black")
        ax1.plot(t, sol[1], linewidth=0.5, color="red")
        ax1.plot(t, sol[2], linewidth=0.5, color="blue")
        plt.show()

