from NN_framework import NN
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Activation_Functions import relu, softmax


X = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
     "X", "Y", "Z"]

nn = NN(layers=[(784, relu), 
                (50, relu), 
                (34, softmax)],
        learning_rate=0.005)


full_data_input = []
full_data_output = []

# nn.load_variables()


data = np.genfromtxt('C:\\Users\\48664\\Desktop\\Projekty\\NerualNetworkFramework\\Neural_Network_Framework\\DataSets\\data_set.csv', delimiter=',')

nn.load_data(data)

nn.learn(150)

# nn.save_variables()

data = np.genfromtxt('C:\\Users\\48664\\Desktop\\Projekty\\NerualNetworkFramework\\Neural_Network_Framework\\DataSets\\data_set_check.csv', delimiter=',')

nn.load_data(data)

print("Start testing: ")
check_data = np.zeros(34)
temp = 0
for i in range(len(nn.full_data_input)):
    nn.training_inputs[0] = nn.full_data_input[i]
    out = nn.full_forward_pass(0)
    check_data[np.argmax(out)] += 1
    if (np.argmax(out)) == np.argmax(nn.full_data_output[i]):
        temp += 1
print("Accuracy: " + str(temp / len(nn.full_data_input)))
print(check_data)
print("Testing on unknown data!!!")

data = np.genfromtxt('C:\\Users\\48664\\Desktop\\Projekty\\NerualNetworkFramework\\Neural_Network_Framework\\DataSets\\new_data_set_test_500.csv', delimiter=',')
nn.load_data(data)
print("Start testing: ")
check_data = np.zeros(34)
temp = 0
for i in range(len(nn.full_data_input)):
    nn.training_inputs[0] = nn.full_data_input[i]
    out = nn.full_forward_pass(0)
    check_data[np.argmax(out)] += 1
    if (np.argmax(out)) == np.argmax(nn.full_data_output[i]):
        temp += 1
print("Accuracy: " + str(temp / len(nn.full_data_input)))
print(check_data)

