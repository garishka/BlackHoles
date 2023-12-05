import numpy as np
from typing import Callable, Union, List
import warnings
from numba.experimental import jitclass
import numba


# Този клас е практически същият като от статията - просто си добавих docstring и преименувах неща, за да ми е
# по-лесно да разбера какво правя.
# FANTASY: User-friendly Symplectic Geodesic Integrator for Arbitrary Metrics with Automatic Differentiation
# https://doi.org/10.3847/1538-4357/abdc28
#@jitclass([("a", numba.float64), ("b", numba.float64)])
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


class HyperDual:
    """
        Support for hyper-dual numbers.

        Attributes:
        -------------
        - a0 (float): The primary real number component.
        - a1 (float, optional): The coefficient of the first-order infinitesimal term (default is 0.).
        - a2 (float, optional): The coefficient of the second-order infinitesimal term (default is 0.).
        - a3 (float, optional): The coefficient of the non-vanishing second-order infinitesimal term (default is 0.).

        Methods:
        ------------
        - __mul__: Multiply with another HyperDual number or scalar.
        - __rmul__: Right multiplication with a scalar.
        - __add__: Add another HyperDual number or scalar.
        - __radd__: Right addition with a scalar.
        - __sub__: Subtract another HyperDual number or scalar.
        - __rsub__: Right subtraction with a scalar.
        - __truediv__: Divide by another HyperDual number or scalar.
        - __rtruediv__: Right division by a scalar.
        - __neg__: Negate the HyperDual number.
        - __pow__: Compute the power of the HyperDual number.
        - sin: Compute the sine of the HyperDual number.
        - cos: Compute the cosine of the HyperDual number.
        - tan: Compute the tangent of the HyperDual number.
        - log: Compute the natural logarithm of the HyperDual number.
        - exp: Compute the exponential of the HyperDual number.
        """
    def __init__(self, a0, a1=0., a2=0., a3=0.):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def __mul__(self, num):
        if isinstance(num, HyperDual):
            return HyperDual(
                self.a0 * num.a0,
                self.a0 * num.a1 + self.a1 * num.a0,
                self.a0 * num.a2 + self.a2 * num.a0,
                self.a0 * num.a3 + self.a1 * num.a2 + self.a2 * num.a1 + self.a3 * num.a0
            )
        else:
            return HyperDual(self.a0 * num, self.a1 * num, self.a2 * num, self.a3 * num)

    def __rmul__(self, num):
        return self.__mul__(num)

    def __add__(self, num):
        if isinstance(num, HyperDual):
            return HyperDual(self.a0 + num.a0, self.a1 + num.a1, self.a2 + num.a2, self.a3 + num.a3)
        else:
            return HyperDual(self.a0 + num, self.a1, self.a2, self.a3)

    def __radd__(self, num):
        return self.__add__(num)

    def __sub__(self, num):
        if isinstance(num, HyperDual):
            return HyperDual(self.a0 - num.a0, self.a1 - num.a1, self.a2 - num.a2, self.a3 - num.a3)
        else:
            return HyperDual(self.a0 - num, self.a1, self.a2, self.a3)

    def __rsub__(self, num):
        return HyperDual(num, 0, 0, 0) - self

    def __truediv__(self, num):
        if isinstance(num, HyperDual):
            if num.a0 != 0:
                inv_num = HyperDual(
                    1.0 / num.a0,
                    - num.a1 / num.a0 ** 2,
                    - (num.a2 / num.a0) ** 2,
                    num.a3 / num.a0 ** 2 - 2 * num.a1 * num.a2 / num.a0 ** 3
                )
                return HyperDual(self.a0 / inv_num.a0, self.a1 / inv_num.a1, self.a2 / inv_num.a2, self.a3 / inv_num.a3)
        else:
            return HyperDual(self.a0 / num, self.a1 / num, self.a2 / num, self.a3 / num)

    def __rtruediv__(self, num):
        return HyperDual(num, 0, 0, 0).__truediv__(self)

    def __neg__(self):
        return HyperDual(-self.a0, -self.a1, -self.a2, -self.a3)

    def __pow__(self, p):
        return HyperDual(self.a0 ** p,
                         self.a1 * p * self.a0 ** (p - 1),
                         self.a2 * p * self.a0 ** (p - 1),
                         self.a3 * p * self.a0 ** (p - 1) + self.a1 * self.a2 * p * (p - 1) * self.a0 ** (p - 2))

    def sin(self):
        return HyperDual(
            np.sin(self.a0),
            self.a1 * np.cos(self.a0),
            self.a2 * np.cos(self.a0),
            self.a3 * np.cos(self.a0) - self.a1 * self.a2 * np.sin(self.a0)
        )

    def cos(self):
        return HyperDual(
            np.cos(self.a0),
            -self.a1 * np.sin(self.a0),
            -self.a2 * np.sin(self.a0),
            -self.a3 * np.sin(self.a0) - self.a1 * self.a2 * np.cos(self.a0)
        )

    def tan(self):
        return self.sin() / self.cos()

    def log(self):
        return HyperDual(
            np.log(self.a0),
            self.a1 / self.a0,
            self.a2 / self.a0,
            self.a3 / self.a0 - (self.a1 * self.a2) / self.a0**2
        )

    def exp(self):
        exp_val = np.exp(self.a0)
        return HyperDual(
            exp_val,
            self.a1 * exp_val,
            self.a2 * exp_val,
            self.a3 * exp_val + self.a1 * self.a2 * exp_val
        )


#@numba.njit
def derivative(func: Callable, x: float):
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
        """
    dual_x = DualNumber(x, 1.)
    dual_func = func(dual_x)
    if isinstance(dual_func, DualNumber):
        return dual_func.b
    else:
        return 0


#@numba.njit
def partial_deriv(func: Callable, vars: Union[list, np.ndarray], wrt_index: int, *params):
    """
    Compute the partial derivative of a multivariable function with respect to a specific variable.

    Parameters:
    -----------
    func : callable
        The multivariable function to be differentiated.
    vars : list
        List of variables for the multivariable function.
    wrt_index : int
        Index of the variable with respect to which the partial derivative is computed.
    *params : float
        Additional parameters to be passed to the multivariable function.

    Returns:
    --------
    float
        The value of the partial derivative at the specified point.

    Notes:
    ------
    The input function should have the form f(vars, params).

    Example:
    ---------
    >>> def multivariable_function(variables: Union[list, np.ndarray]):
    ...     x, y, z = variables
    ...     return x**2 + y**3 - z
    >>> vars = [2.0, 1.0, 3.0]
    >>> wrt_index = 1
    >>> partial_deriv(multivariable_function, vars, wrt_index)
    3.0
    """
    dual_vars = [DualNumber(i, 0) for i in vars]
    dual_vars[wrt_index].b = 1.

    if len(params) != 0:
        dual_func = func(dual_vars, *params)
    else:
        dual_func = func(dual_vars)

    if isinstance(dual_func, DualNumber):
        return dual_func.b
    else:
        return 0


def jacobian(new_coord: List[Callable], old_coord: Union[list, np.ndarray], *params):
    """
        Compute the Jacobian matrix for a set of functions representing new coordinates with respect to
        a set of functions representing old coordinates.

        Parameters:
        -----------
        new_coord : List[Callable]
            List of functions representing the new coordinates.
        old_coord : Union[list, np.ndarray]
            List or array of floats representing the old coordinates.
        *params : float
            Additional parameters to be passed to the functions representing the new coordinates.

        Returns:
        --------
        np.ndarray
            The Jacobian matrix representing the partial derivatives of new coordinates with respect to old coordinates.

        Example:
        ---------
        >>> def coord_func1(vars):
        ...     x, y = vars
        ...     return x**2 + y**2
        >>> def coord_func2(vars):
        ...     x, y = vars
        ...     return x - y
        >>> new_coord = [coord_func1, coord_func2]
        >>> old_coord = [2.0, 1.0]
        >>> jacobian(new_coord, old_coord)
        array([[ 4.,  2.],
               [ 1., -1.]])
        """
    n = len(new_coord)
    m = len(old_coord)

    j = np.zeros(shape=(n, m), dtype=float)

    for i in range(n):
        for k in range(m):
            j[i, k] = partial_deriv(new_coord[i], old_coord, k, *params)

    return j


def second_deriv(func: Callable, x: float, *params):
    hd_x = HyperDual(x, 1., 1., 1.)

    if len(params) != 0:
        hd_func = func(hd_x, params)
    else:
        hd_func = func(hd_x)

    if isinstance(hd_func, HyperDual):
        return (hd_func.a3 - (hd_x.a3 / hd_x.a1) * hd_func.a1) / (hd_x.a1 * hd_x.a2)
    else:
        return 0


def second_partial_deriv(func: Callable, vars: Union[list, np.ndarray], wrt_index: Union[list, np.ndarray], *params):
    global hd_vars
    if len(vars) == 1:
        return second_deriv(func, vars[0], *params)
    elif len(vars) > 1:
        if len(wrt_index) == 2:
            hd_vars = [HyperDual(i, 0, 0, 0) for i in vars]
            if wrt_index[0] != wrt_index[1]:
                hd_vars[wrt_index[0]] = HyperDual(hd_vars[wrt_index[0]].a0, 1., 0., 0.)
                hd_vars[wrt_index[1]] = HyperDual(hd_vars[wrt_index[1]].a0, 0., 1., 0.)
            else:
                hd_vars[wrt_index[0]] = HyperDual(hd_vars[wrt_index[0]].a0, 1., 1., 0.)
                hd_vars[wrt_index[1]] = HyperDual(hd_vars[wrt_index[1]].a0, 1., 1., 0.)
        else:
            raise ValueError("Input list 'wrt_index' must have a length of 2.")

    if len(params) != 0:
        hd_func = func(hd_vars, *params)
    else:
        hd_func = func(hd_vars)

    if isinstance(hd_func, HyperDual):
        return hd_func.a3
    else:
        return 0.

def jacobian_H(H, qp, *params):
    a = params[0]

    j = np.zeros(shape=(8, 8))

    # да поправя реда на диференциране
    for i in range(8):
        for k in range(4):
            j[k, i] = second_partial_deriv(H, qp, [k, i], a)
        for k in range(4, 8):
            j[k, i] = - second_partial_deriv(H, qp, [k, i], a)

    return j

