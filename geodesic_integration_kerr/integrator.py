# TODO: set up на уравненията на Хамилтън във форма, удобна за метода
# TODO: да напиша интегратора
import numpy as np
import warnings


# Този клас е практически същият като от статията - просто си добавих docstring и преименувах неща, за да ми е
# по-лесно да разбера какво правя.
# FANTASY: User-friendly Symplectic Geodesic Integrator for Arbitrary Metrics with Automatic Differentiation
# https://doi.org/10.3847/1538-4357/abdc28
class DualNumber:
    """
    This class provides support for dual numbers, which are expressions of the form a + b * ε,
    where a and b are real numbers, and ε is an infinitesimal dual unit.

    Using dual numbers simplifies the calculation of derivatives for functions.

    Attributes:
    ----------
    a : float
        The real part of the dual number.
    b : float
        The dual part of the dual number, representing the infinitesimal dual unit.

    Methods:
    ----------
    __mul__(self, num)
        Defines multiplication of dual numbers.

    __rmul__(self, num)
        Defines the right multiplication of dual numbers.

    __add__(self, num)
        Defines addition for two dual numbers.

    __radd__(self, num)
        Defines the right addition of dual numbers.

    __sub__(self, num)
        Defines subtraction for two dual numbers.

    __rsub__(self, num)
        Defines the right subtraction of dual numbers.

    __truediv__(self, num)
        Defines true division for two dual numbers.

    __rtruediv__(self, num)
        Defines the right true division of dual numbers.

    __neg__(self)
        Returns the negative of a dual number.

    __pow__(self, power)
        Raises a dual number to a power.

    sin(self)
        Returns a new DualNumber representing the sine of the original DualNumber.

    cos(self)
        Returns a new DualNumber representing the cosine of the original DualNumber.

    tan(self)
        Returns a new DualNumber representing the tangent of the original DualNumber.

    log(self)
        Returns a new DualNumber representing the natural logarithm of the original DualNumber.

    exp(self)
        Returns a new DualNumber representing the exponential function of the original DualNumber.
    """
    def __init__(self, a: float, b: float):
        """
        Constructor

        Parameters
        ----------
        :param a: float
            First component of the dual number of the form a + b * dx
        :param b: float
            Second component of the dual number.
        """
        self.a = a
        self.b = b

    def __mul__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a * num.a, self.b * num.a + self.a * num.b)
        else:
            return DualNumber(self.a * num, self.b * num)

    def __rmul__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a * num.a, self.b * num.a + self.a * num.b)
        else:
            return DualNumber(self.a * num, self.b * num)

    def __add__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a + num.a, self.b + num.b)
        else:
            return DualNumber(self.a + num, self.b)

    def __radd__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a + num.a, self.b + num.b)
        else:
            return DualNumber(self.a + num, self.b)

    def __sub__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a - num.a, self.b - num.b)
        else:
            return DualNumber(self.a - num, self.b)

    def __rsub__(self, num):
        return DualNumber(num, 0) - self

    def __truediv__(self, num):
        if isinstance(num, DualNumber):
            return DualNumber(self.a / num.a, (self.b * num.a - self.a * num.b) / (num.a ** 2.))
        else:
            return DualNumber(self.a / num, self.b / num)

    def __rtruediv__(self, num):
        return DualNumber(num, 0).__truediv__(self)

    def __neg__(self):
        return DualNumber(-self.a, -self.b)

    def __pow__(self, power):
        return DualNumber(self.a ** power, self.b * power * self.a ** (power - 1))

    def sin(self):
        return DualNumber(np.sin(self.a), self.b * np.cos(self.a))

    def cos(self):
        return DualNumber(np.cos(self.a), -self.b * np.sin(self.a))

    def tan(self):
        return self.sin() / self.cos()

    def log(self):
        return DualNumber(np.log(self.a), self.b / self.a)

    def exp(self):
        return DualNumber(np.exp(self.a), self.b * np.exp(self.a))


def derivative(func, x):
    """
        Calculate the derivative of a given function at a specific point using dual numbers.

        Parameters:
        -----------
        func : callable
            The function for which the derivative is to be calculated.
        x : float
            The point at which to evaluate the derivative.

        Returns:
        -----------
        float
            The derivative of the function at the specified point.
            If the function can be represented as a dual number, the result is the dual part (b) of the dual number.
            If the function is a constant, the derivative is always zero, and the function returns 0.
        """
    dual_x = DualNumber(x, 1.)
    dual_func = func(dual_x)
    if isinstance(dual_func, DualNumber):
        return func(dual_x).b
    else:
        # function is constant
        return 0


class SymplecticIntegrator:
    def __init__(self, q0, p0, metric_params, steps, order, null=True):
        self.q0 = q0
        self.p0 = p0
        self.metric_params = metric_params
        self.steps = steps
        self.null = null
        self.order = order


