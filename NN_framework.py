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

    def add_layer(self, inputs, neurons, last_layer, first_layer):
        self.layers.append(Layer(inputs, neurons, last_layer, first_layer))

    def full_forward_pass(self, num_of_train):
        input_for_next_layer = self.training_inputs[num_of_train]
        for layer_num in range(len(self.layers)):
            input_for_next_layer = self.layers[layer_num].forward_propagation(input_for_next_layer)
        return input_for_next_layer

    def full_backward_pass(self, num_of_training):
        for layer_num in range(len(self.layers) - 1, -1, -1):
            if layer_num - 1 >= 0:
                prev_layer = nn.layers[layer_num - 1]
            else:
                prev_layer = None
            if layer_num + 1 < len(self.layers):
                next_layer = nn.layers[layer_num+1]
            else:
                next_layer = None
            nn.layers[layer_num].back_propagation(nn.training_outputs[num_of_training],
                                                  nn.training_inputs[num_of_training], prev_layer, next_layer)

    def update_all(self):
        for layer_num in range(len(self.layers)):
            self.layers[layer_num].update_values(self.learning_rate)


nn = NN(0.05)

nn.add_layer(784, 15, False, True)
nn.add_layer(15, 10, True, False)

full_data_input = []
full_data_output = []

with open('train.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    count = 0
    for row in csv_reader:
        if count != 0:
            full_data_input.append([])
            full_data_output.append([])
            for i in range(0, 10):
                if int(row[0]) == i:
                    full_data_output[count - 1].append(1)
                else:
                    full_data_output[count - 1].append(0)
            for i in range(1, len(row)):
                full_data_input[count - 1].append(int(row[i]) / 255)
            if count == 1000:
                break
        count += 1

start_time = time.time()
for j in range(30):
    temp = 0
    for i in range(0, 1000):
        nn.training_inputs[0] = full_data_input[i]
        nn.training_outputs[0] = full_data_output[i]
        prediction = nn.full_forward_pass(0)
        if np.argmax(prediction) == np.argmax(full_data_output[i]):
            temp += 1
        nn.full_backward_pass(0)
        nn.update_all()
    print("Accuracy: " + str(temp/1000))

print("\n")
nn.training_inputs[0] = full_data_input[10]
out = nn.full_forward_pass(0)

zzz = np.array(nn.training_inputs[0]).reshape((28, 28)) * 255

plt.gray()
plt.imshow(zzz, interpolation='nearest')
plt.show()
print(out*100)
print(np.argmax(out))
print(full_data_output[10])

print("\n")

# nn.training_inputs[0] = full_data_input[8]
# out = nn.full_forward_pass(0)
# print(out)
# print(np.argmax(out))
# print(full_data_output[8])
#
# print("\n")
#
# nn.training_inputs[0] = full_data_input[66]
# out = nn.full_forward_pass(0)
# print(np.argmax(out))
# print(full_data_output[66])
#
# print("\n")
#
# nn.training_inputs[0] = full_data_input[74]
# out = nn.full_forward_pass(0)
# print(np.argmax(out))
# print(full_data_output[74])
#
# print("\n")

print("--- %s seconds ---" % (time.time() - start_time))
