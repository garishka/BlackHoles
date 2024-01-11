import numba
from typing import Callable
import numpy as np


@numba.jit(nopython=True)
def foo(func: Callable, params: np.ndarray, const: float, args: float):
    res = const
    for param in params:
        res += func(param, args)
    return res


@numba.njit
def square(x: float, a: float):
    return a * x ** 2


params_array = np.array([1.0, 2.0, 3.0])
constant_value = 10.0

result = foo(square, params_array, constant_value, 2.0)
print(result)
