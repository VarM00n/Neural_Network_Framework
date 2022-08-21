from random import randint
from VMNeuralNetwork.NN_framework import NN
from VMNeuralNetwork.Activation_Functions import relu, softmax
import numpy as np
import time
def lin_func(a, b, x):
    return a*x + b


def is_below(a, b, xp, yp):
    return yp - (a * xp + b) < 0 or yp - lin_func(a, b, xp) < 0


data_set = []

for i in range(10000):
    xp, yp = randint(-100, 000), randint(-100, 100)
    data_set.append([is_below(2, 3, xp, yp), xp, yp])
nn = NN(layers=[(2, relu),
                (10, relu),
                (2, softmax)],
        learning_rate=0.0006)
nn.load_data_with_output(data_set)
nn.learn(25)

data_set = []
for i in range(100):
    xp, yp = randint(-100000, 100000), randint(-100000, 100000)
    data_set.append([xp, yp])

nn.load_data(data_set)
print(len(nn.full_data_input))
check_data = np.zeros(2)
temp = 0
start_time = time.time()
for i in range(len(nn.full_data_input)):
    # nn.training_inputs[0] = nn.full_data_input[i]
    out = nn.predict(i)
    # print(np.argmax(nn.full_data_output[i]), np.argmax(out))
    # if (np.argmax(out)) == np.argmax(nn.full_data_output[i]):
    #     temp += 1
        
print((time.time() - start_time)/ 10000)
# print("Accuracy: " + str(temp / len(nn.full_data_input)))
# print(check_data)
