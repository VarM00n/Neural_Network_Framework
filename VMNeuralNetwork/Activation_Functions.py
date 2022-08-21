import numpy as np
import scipy
from typing import Union, Callable


def relu(array: list[Union[int, float]] = None, 
         x: float = None) -> float:
    return 0 if x < 0 else x

def relu_deriv(x: float) -> Union[0, 1]:
    return 0 if x < 0 else 1

def softmax(array: list[Union[int, float]] = None,
            x: float = None) -> list[float]:
    return scipy.special.softmax(array)[np.where(array == x)[0][0]]
