from NN_framework import NN
from NN_framework import NN
# from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt
import csv

# mndata = MNIST("letters")

X = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"]

nn = NN(0.1)  # initializing neural network with learning rate as a param

nn.add_layer(1023, 50, False, True, 1)  # prev_layer_neurons, curr_layer_neurons, isLastLayer, isFirstLayer
nn.add_layer(50, 5, True, False, 2)

# nn.load_variables()

full_data_input = []

with open('test3.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    count = 0
    for row in csv_reader:
        full_data_input.append(row[1:len(row)-1])
    print("done")

full_data_output = []
with open('output.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    count = 0
    for row in csv_reader:
        full_data_output.append(row)
    print("done")
full_data_input = np.asarray(full_data_input)
full_data_input = full_data_input.astype(float)
full_data_output = np.asarray(full_data_output)
full_data_output = full_data_output.astype(float)

new_training_set = []
new_output_set = []
for i in range(700):
    index = random.randrange(0, len(full_data_input))
    new_training_set.append(full_data_input[index])
    new_output_set.append(full_data_output[index])

accuracy = 0
l = 0
for i in range(20):
    l += 1
    temp = 0
    for i in range(0, len(new_training_set)):
        nn.training_inputs[0] = new_training_set[i]
        nn.training_outputs[0] = new_output_set[i]
        prediction = nn.full_forward_pass(0)
        if np.argmax(prediction) == np.argmax(new_output_set[i]):
            temp += 1
        nn.full_backward_pass(0)
        nn.update_all()
    accuracy = temp/len(new_training_set)
    print("Accuracy: " + str(temp/len(new_training_set)))
    if accuracy == 1:
        break
print("Epochs: " + str(l))

# images, labels = mndata.load_testing()
# full_data_input = []
# for i in range(len(images)):
#     full_data_output.append([])
#     full_data_input.append([])
#     full_data_input[i] = images[i]
#     full_data_input[i] = np.array(full_data_input[i])
#     full_data_input[i] = full_data_input[i] / 255
#     # print(labels[i])
#     for j in range(0, 26):
#         if j == labels[i]:
#             full_data_output[i].append(1)
#         else:
#             full_data_output[i].append(0)
    # print(full_data_output[i])
#
print("Start testing: ")
temp = 0
for i in range(len(full_data_input)):
    index = random.randrange(0, len(full_data_input))
    a = input()
    # print(i)
    nn.training_inputs[0] = full_data_input[index]
    out = nn.full_forward_pass(0)
    zzz = np.array(nn.training_inputs[0]).reshape((28, 28)) * 255
    zzz = zzz.T
    plt.gray()
    plt.imshow(zzz, interpolation='nearest')
    plt.show()
    print("Prediction: " + str(np.argmax(out)))
    print("Actual: " + str(np.argmax(full_data_output[index])))
    if (np.argmax(out) + 1) == np.argmax(full_data_output[index]):
        temp += 1
print("Accuracy: " + str(temp / len(new_training_set)))
