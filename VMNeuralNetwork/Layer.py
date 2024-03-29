from typing import Union, Callable, Type
import numpy as np
from .Activation_Functions import *


class Layer:

    dz = []
    dw = []
    db = []

    def __init__(self, 
                 inputs: int, 
                 neurons: int, 
                 activation_function: Callable, 
                 last_layer: bool, 
                 first_layer: bool, 
                 id_num) -> None:
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(neurons, inputs))
        self.bias = np.ones(neurons)
        self.z = np.zeros(neurons)
        self.dz = np.zeros(neurons)
        self.output = np.zeros(neurons)
        self.activation_function = activation_function
        self.last_layer = last_layer
        self.first_layer = first_layer
        self.neurons = neurons
        self.id_num = id_num

    def forward_propagation(self, 
                            input_values: list[Union[float, int]]) -> list[float]:
        self.z = self.weights.dot(input_values) + self.bias
        self.output = np.array([self.activation_function(self.z, x) for x in self.z])
        return self.output

    def back_propagation(self, 
                         training_output: list[Union[float, int]], 
                         training_input: list[Union[float, int]], 
                         prev_layer: 'Layer' = None, 
                         next_layer: 'Layer' = None) -> None:
        # Calculation of dz
        if self.activation_function == softmax:
            self.dz = self.output - training_output
        else:
            transformed_weights = next_layer.weights.T
            temp2 = next_layer.dz.reshape((-1, 1))
            self.dz = transformed_weights.dot(temp2).T * np.array([relu_deriv(x) for x in self.z])
        # Calculation of dw
        if self.first_layer:
            self.dw = self.dz.reshape((-1, 1)) * training_input
        else:
            self.dw = self.dz.reshape((-1, 1)) * prev_layer.output
        # Calculation of db
        self.db = self.dz        

    def update_values(self, 
                      learning_rate: Union[int, float]) -> None:
        self.weights = self.weights - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db
        self.bias = self.bias.reshape(self.neurons, )

    def save_to_file(self) -> None:
        file_name_weights = "savedWages/wages/w_" + str(self.id_num) + ".csv"
        file_name_biases = "savedWages/biases/b_" + str(self.id_num) + ".csv"
        np.savetxt(file_name_weights, self.weights, delimiter=',')
        np.savetxt(file_name_biases, self.bias, delimiter=',')

    def load_from_file(self) -> None:
        file_name_weights = "savedWages/wages/w_" + str(self.id_num) + ".csv"
        file_name_biases = "savedWages/biases/b_" + str(self.id_num) + ".csv"
        self.weights = np.loadtxt(file_name_weights, delimiter=',')
        self.bias = np.loadtxt(file_name_biases, delimiter=',')
