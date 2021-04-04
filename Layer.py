import numpy as np

from Activation_Functions import *

# TODO complete activation functions
# TODO create Neural Network class


class Layer:

    dz = []
    dw = []
    db = []

    def __init__(self, inputs, neurons, last_layer):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(neurons, inputs))
        self.bias = np.ones(neurons)
        self.z = np.zeros(neurons)
        self.dz = np.zeros(neurons)
        self.output = np.zeros(neurons)
        self.last_layer = last_layer
        self.neurons = neurons

    def forward_propagation(self, input_values):
        self.z = np.zeros(self.neurons)
        self.z = self.weights.dot(input_values) + self.bias
        if self.last_layer:
            self.output = np.array([softmax(self.z, x) for x in self.z])
        else:
            self.output = np.array([relu(x) for x in self.z])
        return self.output

    # TODO change updating dw with training_input (not always the same)
    def back_propagation(self, training_output, training_input, prev_layer=None, next_layer=None):
        if self.last_layer:
            self.dz = self.output - training_output
            temp = self.dz.reshape((-1, 1))
            self.dw = temp * prev_layer.output
            self.db = self.dz
        else:
            temp1 = next_layer.weights.T
            temp2 = next_layer.dz.reshape((-1, 1))
            self.dz = temp1.dot(temp2).T * np.array([relu_deriv(x) for x in self.z])
            self.dw = self.dz.reshape((-1, 1)) * training_input
            self.db = self.dz

    def update_values(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db
        self.bias = self.bias.reshape(self.neurons, )
