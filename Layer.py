import numpy as np

from Activation_Functions import *

# TODO complete activation functions
# TODO create Neural Network class


class Layer:

    def __init__(self, inputs, neurons, last_layer):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(neurons, inputs))
        self.bias = np.ones(neurons)
        self.z = np.zeros(neurons)
        self.output = np.zeros(neurons)
        self.last_layer = last_layer

    def forward_propagation(self, input_values):
        self.z = self.weights.dot(input_values) + self.bias
        if self.last_layer:
            self.output = np.array([softmax(self.z, x) for x in self.z])
        else:
            self.output = np.array([relu(x) for x in self.z])
        return self.output




