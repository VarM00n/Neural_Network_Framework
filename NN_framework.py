from Layer import Layer
import numpy as np
import csv
import time
from matplotlib import pyplot as plt
import math


class NN:
    layers = []
    training_inputs = [[]]
    training_outputs = [[]]
    learning_rate = 0

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def add_layer(self, inputs, neurons, last_layer, first_layer, id):
        self.layers.append(Layer(inputs, neurons, last_layer, first_layer, id))

    def full_forward_pass(self, num_of_train):
        input_for_next_layer = self.training_inputs[num_of_train]
        for layer_num in range(len(self.layers)):
            input_for_next_layer = self.layers[layer_num].forward_propagation(input_for_next_layer)
        return input_for_next_layer

    def full_backward_pass(self, num_of_training):
        for layer_num in range(len(self.layers) - 1, -1, -1):
            if layer_num - 1 >= 0:
                prev_layer = self.layers[layer_num - 1]
            else:
                prev_layer = None
            if layer_num + 1 < len(self.layers):
                next_layer = self.layers[layer_num + 1]
            else:
                next_layer = None
            self.layers[layer_num].back_propagation(self.training_outputs[num_of_training],
                                                    self.training_inputs[num_of_training], prev_layer, next_layer)

    def update_all(self):
        for layer_num in range(len(self.layers)):
            self.layers[layer_num].update_values(self.learning_rate)

    def save_variables(self):
        for layer in self.layers:
            layer.save_to_file()

    def load_variables(self):
        for layer in self.layers:
            layer.load_from_file()
