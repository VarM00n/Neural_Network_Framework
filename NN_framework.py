from Layer import Layer
import numpy as np
import csv


class NN:
    layers = []
    training_inputs = [[0.3, 0.1, 0.4]]
    training_outputs = [[1, 0]]
    learning_rate = 0

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def add_layer(self, inputs, neurons, last_layer):
        self.layers.append(Layer(inputs, neurons, last_layer))

    def full_forward_pass(self, num_of_train):
        input_for_next_layer = self.training_inputs[num_of_train]
        for i in range(len(self.layers)):
            input_for_next_layer = self.layers[i].forward_propagation(input_for_next_layer)
        return input_for_next_layer

    def full_backward_pass(self, num_of_training):
        for i in range(len(self.layers) - 1, -1, -1):
            if i - 1 >= 0:
                prev_layer = nn.layers[i - 1]
            else:
                prev_layer = None
            if i + 1 < len(self.layers):
                next_layer = nn.layers[i+1]
            else:
                next_layer = None
            nn.layers[i].back_propagation(nn.training_outputs[num_of_training], nn.training_inputs[num_of_training],
                                          prev_layer, next_layer)

    def update_all(self):
        for i in range(len(self.layers)):
            self.layers[i].update_values(self.learning_rate)


nn = NN(0.2)

nn.add_layer(784, 10, False)
nn.add_layer(10, 10, True)
nn.full_forward_pass(0)
nn.full_backward_pass(0)
nn.update_all()
print("done")
