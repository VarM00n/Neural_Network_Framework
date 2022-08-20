import numpy as np
import math
import scipy

def relu(x):
    return 0 if x < 0 else x

def relu_deriv(x):
    return 0 if x < 0 else 1

def softmax(array, x):
    return scipy.special.softmax(array)[np.where(array == x)[0][0]]
