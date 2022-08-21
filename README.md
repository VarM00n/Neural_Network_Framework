# VM Neural Network
## Description
Neural network framework for learning purposes. It allows to implement feedforward neural network with fixed number of layers and neurons. 
## Download
To download a package please use a command provided below:
```
pip install VMNeuralNetwork
```
## Tutorial
### Creating test dataset
Dataset that will be used in a tutorial is a set of points `[(x, y), ...]` and the problem is to determin whether tha point lies above or below the line.
Firstly define functions for calculating:
  - function value
    ```
    def lin_func(a, b, x):
      return a*x + b
    ```
  - point position
    ```
    def is_below(a, b, xp, yp):
      return yp - (a * xp + b) < 0 or yp - lin_func(a, b, xp) < 0
    ```
Next step is to generate test dataset (10000 sets with a structure - `(bolow - 0|above - 1, x, y)`
```
from random import randint

for i in range(10000):
    xp, yp = randint(-100, 000), randint(-100, 100)
    data_set.append([is_below(2, 3, xp, yp), xp, yp])
```
### Initialazing neural network
```
from VMNeuralNetwork.NN_framework import NN
from VMNeuralNetwork.Activation_Functions import relu, softmax
nn = NN(layers=[(2, relu),
                (10, relu),
                (2, softmax)],
        learning_rate=0.0006)
```
### Loading dataset and training process
Two functions has been created for loading data, one, used below takes data with etiquette and the second one without.
```
nn.load_data_with_output(data_set)
nn.learn(25)
```
By this point you have succesfully created and trained neural network. Second step is to check if it properly predicts values. It is being done by taking subarray of dataset, loading that data and predicting.
```
data_set = [data_set[randint(0, 9999)] for i in range(1000)]
nn.load_data_with_output(data_set)
correct_predictions = 0
for i in range(len(data_set)):
    out = nn.predict(i)
    if np.argmax(out) == data_set[i][0]:
        correct_predictions += 1
print("Final accuracy: " + str(correct_predictions / len(data_set)))
```
Last step is to try out trained network using freshly generated data. The process stays the same as previously - loading data -> predicting.
```
for i in range(20):
    xp, yp = randint(-100, 100), randint(-100, 100)
    data_set.append([xp, yp])
nn.load_data(data_set)
for i in range(len(data_set)):
    out = nn.predict(i)
    print(f'x - {data_set[i][0]}, y - {data_set[i][1]}\t--\tprediction - {out}')
```