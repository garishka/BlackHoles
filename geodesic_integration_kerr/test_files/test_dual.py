import unittest
from geodesic_integration_kerr import dual
import numpy as np
import random


class TestPartialDerivative(unittest.TestCase):

    def test_partial_deriv1(self):
        def f1(vars):
            x, y, z = vars
            return x ** 2 + y ** 2 - z ** 2

        def df1dx(vars):
            x, y, z = vars
            return 2 * x

        def df1dy(vars):
            x, y, z = vars
            return 2 * y

        def df1dz(vars):
            x, y, z = vars
            return - 2 * z

        nums = [0, 1, -1, 10, 5, 6, -100, 46]

        for i in range(10):
            x1 = nums[random.randint(0, len(nums) - 1)]
            y1 = nums[random.randint(0, len(nums) - 1)]
            z1 = nums[random.randint(0, len(nums) - 1)]
            var = [x1, y1, z1]

            dfdx = df1dx(var)
            self.assertEqual(dual.partial_deriv(f1, var, 0), dfdx)

            dfdy = df1dy(var)
            self.assertEqual(dual.partial_deriv(f1, var, 1), dfdy)

            dfdz = df1dz(var)
            self.assertEqual(dual.partial_deriv(f1, var, 2), dfdz)

    def test_partial_deriv2(self):
        def f2(vars):
            x, y, z = vars
            return 2 * x * np.sin(y) * np.cos(z)

        def df2dx(vars):
            x, y, z = vars
            return 2 * np.sin(y) * np.cos(z)

        def df2dy(vars):
            x, y, z = vars
            return 2 * x * np.cos(y) * np.cos(z)

        def df2dz(vars):
            x, y, z = vars
            return - 2 * x * np.sin(y) * np.sin(z)

        nums = [0, 1, -1, 10, 5, 6, -100, 46]

        for i in range(10):
            x2 = nums[random.randint(0, len(nums) - 1)]
            y2 = nums[random.randint(0, len(nums) - 1)]
            z2 = nums[random.randint(0, len(nums) - 1)]
            var = [x2, y2, z2]

            dfdx = df2dx(var)
            self.assertEqual(dual.partial_deriv(f2, var, 0), dfdx)

            dfdy = df2dy(var)
            self.assertEqual(dual.partial_deriv(f2, var, 1), dfdy)

            dfdz = df2dz(var)
            self.assertEqual(dual.partial_deriv(f2, var, 2), dfdz)

    def test_partial_deriv3(self):
        def f3(vars):
            x, y, z = vars
            return 5

        nums = [0, 1, -1, 10, 5, 6, -100, 46]

        for i in range(10):
            x3 = nums[random.randint(0, len(nums) - 1)]
            y3 = nums[random.randint(0, len(nums) - 1)]
            z3 = nums[random.randint(0, len(nums) - 1)]
            var = [x3, y3, z3]

            self.assertEqual(dual.partial_deriv(f3, var, 0), 0)

            self.assertEqual(dual.partial_deriv(f3, var, 1), 0)

            self.assertEqual(dual.partial_deriv(f3, var, 2), 0)

    def test_partial_deriv4(self):
        def f4(vars):
            x, y, z = vars
            return x * np.exp(y-z)

        def df4dx(vars):
            x, y, z = vars
            return np.exp(y-z)

        def df4dy(vars):
            x, y, z = vars
            return f4(vars)

        def df4dz(vars):
            x, y, z = vars
            return - f4(vars)

        nums = [0, 1, -1, 10, 5, 6, -100, 46]

        for i in range(10):
            x4 = nums[random.randint(0, len(nums) - 1)]
            y4 = nums[random.randint(0, len(nums) - 1)]
            z4 = nums[random.randint(0, len(nums) - 1)]
            var = [x4, y4, z4]

            dfdx = df4dx(var)
            self.assertEqual(dual.partial_deriv(f4, var, 0), dfdx)

            dfdy = df4dy(var)
            self.assertEqual(dual.partial_deriv(f4, var, 1), dfdy)

            dfdz = df4dz(var)
            self.assertEqual(dual.partial_deriv(f4, var, 2), dfdz)

    def test_partial_deriv5(self):
        def f5(vars, params):
            x, y, z = vars
            a, b = params
            return 2 * a * x**2 + np.sin(b * y) + z

        def df5dx(vars, params):
            x, y, z = vars
            a, b = params
            return 4 * a * x

        def df5dy(vars, params):
            x, y, z = vars
            a, b = params
            return b * np.cos(b * y)

        def df5dz(vars, params):
            x, y, z = vars
            a, b = params
            return 1

        nums = [random.randint(0, 100) for i in range(10)]

        for i in range(10):
            x5 = nums[random.randint(0, len(nums) - 1)]
            y5 = nums[random.randint(0, len(nums) - 1)]
            z5 = nums[random.randint(0, len(nums) - 1)]
            a5 = nums[random.randint(0, len(nums) - 1)]
            b5 = nums[random.randint(0, len(nums) - 1)]
            var = [x5, y5, z5]
            param = [a5, b5]

            dfdx = df5dx(var, param)
            self.assertEqual(dual.partial_deriv(f5, var, 0, param), dfdx)

            dfdy = df5dy(var, param)
            self.assertEqual(dual.partial_deriv(f5, var, 1, param), dfdy)

            dfdz = df5dz(var, param)
            self.assertEqual(dual.partial_deriv(f5, var, 2, param), dfdz)


class TestJacobian(unittest.TestCase):
    def test_polar_to_cartesian(self):
        r = np.linspace(0, 100)
        phi = np.linspace(0, np.pi)

        def x(vars):
            rho, th = vars
            return rho * np.cos(th)
        def y(vars):
            rho, th =vars
            return rho * np.sin(th)

        for i in range(len(r)):
            expected_jacobian = np.array([
                [np.cos(phi[i]), -r[i] * np.sin(phi[i])],
                [np.sin(phi[i]), r[i] * np.cos(phi[i])]
            ])

            calculated_jacobian = dual.jacobian([x, y], [r[i], phi[i]])

            np.testing.assert_allclose(calculated_jacobian, expected_jacobian, atol=1e-8)

    def test_spherical_to_cartesian(self):
        r = np.linspace(0, 100)
        phi = np.linspace(0, np.pi)
        theta = np.linspace(0, 2 * np.pi)

        def x(vars):
            rho, alpha, beta = vars
            return rho * np.sin(alpha) * np.cos(beta)
        def y(vars):
            rho, alpha, beta = vars
            return rho * np.sin(alpha) * np.sin(beta)
        def z(vars):
            rho, alpha, beta = vars
            return rho * np.cos(alpha)

        for i in range(len(r)):
            expected_jacobian = np.array([
                [np.sin(phi[i]) * np.cos(theta[i]), r[i] * np.cos(phi[i]) * np.cos(theta[i]), -r[i] * np.sin(phi[i]) * np.sin(theta[i])],
                [np.sin(phi[i]) * np.sin(theta[i]), r[i] * np.cos(phi[i]) * np.sin(theta[i]), r[i] * np.sin(phi[i]) * np.cos(theta[i])],
                [np.cos(phi[i]), -r[i] * np.sin(phi[i]), 0]
            ])

            calculated_jacobian = dual.jacobian([x, y, z], [r[i], phi[i], theta[i]])

            np.testing.assert_allclose(calculated_jacobian, expected_jacobian, atol=1e-7)



if __name__ == '__main__':
    unittest.main()
