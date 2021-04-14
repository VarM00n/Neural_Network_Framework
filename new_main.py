from NN_framework import NN
from NN_framework import NN
from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt

mndata = MNIST("letters")

X = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"]

nn = NN(0.001)  # initializing neural network with learning rate as a param

nn.add_layer(784, 300, False, True, 1)  # prev_layer_neurons, curr_layer_neurons, isLastLayer, isFirstLayer
nn.add_layer(300, 26, True, False, 3)

nn.load_variables()

print("done")

images, labels = mndata.load_testing()
full_data_output = []
full_data_input = []
for i in range(len(images)):
    full_data_output.append([])
    full_data_input.append([])
    full_data_input[i] = images[i]
    full_data_input[i] = np.array(full_data_input[i])
    full_data_input[i] = full_data_input[i] / 255
    # print(labels[i])
    for j in range(0, 26):
        if j == labels[i]:
            full_data_output[i].append(1)
        else:
            full_data_output[i].append(0)
    # print(full_data_output[i])

print("Start testing: ")
temp = 0
for i in range(len(full_data_input)):
    index = random.randrange(0, len(images))
    # a = input()
    # print(i)
    nn.training_inputs[0] = full_data_input[index]
    out = nn.full_forward_pass(0)
    # zzz = np.array(nn.training_inputs[0]).reshape((28, 28)) * 255
    # zzz = zzz.T
    # plt.gray()
    # plt.imshow(zzz, interpolation='nearest')
    # plt.show()
    # print("Prediction: " + X[int(np.argmax(out))])
    # print("Actual: " + X[int(np.argmax(full_data_output[index])-1)])
    if (np.argmax(out) + 1) == np.argmax(full_data_output[index]):
        temp += 1
print("Accuracy: " + str(temp / len(full_data_input)))
