import numpy as np
import math


def relu(x):
    return 0 if x < 0 else x


def relu_deriv(x):
    return 0 if x < 0 else 1


def softmax(array, x):
    sum_of_exp = np.sum(np.exp(array))
    return math.exp(x)/sum_of_exp