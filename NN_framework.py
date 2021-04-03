from Layer import Layer


class NN:
    layers = []
    training_inputs = [[0.3, 0.1, 0.4]]
    training_outputs = [[1, 0]]

    def __init__(self):
        pass

    def add_layer(self, inputs, neurons, last_layer):
        self.layers.append(Layer(inputs, neurons, last_layer))

    def full_forward_pass(self, num_of_train):
        input_for_next_layer = self.training_inputs[num_of_train]
        for i in range(len(self.layers)):
            input_for_next_layer = self.layers[i].forward_propagation(input_for_next_layer)
        return input_for_next_layer


nn = NN()


nn.add_layer(3, 2, False)
nn.add_layer(2, 2, True)
print(nn.full_forward_pass(0))
print("ff")