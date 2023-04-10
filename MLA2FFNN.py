import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import random


# This is our class to define nodes in our neural network. First we have a constructor that takes in and saves all
# values we deemed important for a node to hold. Next we have methods to update each of these values individually
# when needed. Lastly, we have methods to calculate inputs, outputs, and error for each node in our network.
class Node:
    def __init__(self, weights, bias, net_input, output, error):
        self.weights = weights
        self.bias = bias
        self.net_input = net_input
        self.output = output
        self.error = error

    def updateWeights(self, weights):
        self.weights = weights

    def updateBias(self, bias):
        self.bias = bias

    def updateInput(self, net_input):
        self.net_input = net_input

    def updateOutput(self, output):
        self.output = output

    def calcNetInput(self, weights, train_row, bias):
        self.net_input = np.dot(weights, train_row) + bias

    def calcOutput(self, net_input):
        self.output = 1 / (1 + np.exp(-net_input))

    def calcOutputError(self, actual, output):
        self.error = output * (1 - output) * (actual - output)

    def calcHiddenError(self, output_error, output_weights, own_output):
        self.error = own_output * (1 - own_output) * (
                (output_error * output_weights[0]) + (output_error * output_weights[1]) + (
                output_error * output_weights[2]) + (output_error * output_weights[3]))


# This is our initialize_Node method. It is responsible for initializing nodes weights and biases upon their creation
def initialize_Node(size):
    for i in range(size):
        if i == 0:
            weights = np.array(random.randint(-1, 1))
        else:
            weights = np.append(weights, random.randint(-1, 1))
    bias = random.randint(-1, 1)
    return weights, bias


# This method is responsible for calculating the change in bias when we need to update a node's bias.
def calcBiasChange(own_error):
    return own_error * 0.05


# This method is responsible for calculating the change in weights when we need to update a node's weights.
def calcWeightChange(outputs, error):
    return 0.05 * error * outputs


# This method is responsible for converting all our hidden layer's output into an array so it easier to use in other
# methods.
def createOutputArr():
    output_output = np.array(h1.output)
    output_output = np.append(output_output, h2.output)
    output_output = np.append(output_output, h3.output)
    output_output = np.append(output_output, h4.output)
    return output_output


# This method is responsible for training our model. It works by having an outer for loop that runs for the number of
# epochs specified and an inner for loop that iterates through each individual training sample. For each sample, it
# starts by calculating the hidden layer's net input followed by their outputs. Next, we calculate our output node's
# input and outputs. Next, calculate our errors for each node and back propagate to update each node's weight and biases
# accordingly.
def trainModel(epochs):
    m, n = x_train.shape
    for k in range(epochs):
        for i in range(m):
            current_sample = x_train[i]
            h1.calcNetInput(h1.weights, current_sample, h1.bias)
            h2.calcNetInput(h2.weights, current_sample, h2.bias)
            h3.calcNetInput(h3.weights, current_sample, h3.bias)
            h4.calcNetInput(h4.weights, current_sample, h4.bias)
            h1.calcOutput(h1.net_input)
            h2.calcOutput(h2.net_input)
            h3.calcOutput(h3.net_input)
            h4.calcOutput(h4.net_input)
            o_input = (h1.output * o1.weights[0]) + (h2.output * o1.weights[1]) + (h3.output * o1.weights[2]) + (
                    h4.output * o1.weights[3]) + o1.bias
            o1.updateInput(o_input)
            o1.calcOutput(o1.net_input)
            o1.calcOutputError(y_train[i], o1.output)
            h1.calcHiddenError(o1.error, o1.weights, h1.output)
            h2.calcHiddenError(o1.error, o1.weights, h2.output)
            h3.calcHiddenError(o1.error, o1.weights, h3.output)
            h4.calcHiddenError(o1.error, o1.weights, h4.output)
            h1_weight_change = calcWeightChange(current_sample, h1.error)
            h1_new_weights = h1.weights + h1_weight_change
            h1.updateWeights(h1_new_weights)
            h2_weight_change = calcWeightChange(current_sample, h2.error)
            h2_new_weights = h2.weights + h2_weight_change
            h2.updateWeights(h2_new_weights)
            h3_weight_change = calcWeightChange(current_sample, h3.error)
            h3_new_weights = h3.weights + h3_weight_change
            h3.updateWeights(h3_new_weights)
            h4_weight_change = calcWeightChange(current_sample, h4.error)
            h4_new_weights = h4.weights + h4_weight_change
            h4.updateWeights(h4_new_weights)
            o1_sample = createOutputArr()
            o1_weight_change = calcWeightChange(o1_sample, o1.error)
            o1_new_weights = o1.weights + o1_weight_change
            o1.updateWeights(o1_new_weights)
            h1_bias_change = calcBiasChange(h1.error)
            h1_new_bias = h1.bias + h1_bias_change
            h1.updateBias(h1_new_bias)
            h2_bias_change = calcBiasChange(h2.error)
            h2_new_bias = h2.bias + h2_bias_change
            h2.updateBias(h2_new_bias)
            h3_bias_change = calcBiasChange(h3.error)
            h3_new_bias = h3.bias + h3_bias_change
            h3.updateBias(h3_new_bias)
            h4_bias_change = calcBiasChange(h4.error)
            h4_new_bias = h4.bias + h4_bias_change
            h4.updateBias(h4_new_bias)
            o1_bias_change = calcBiasChange(o1.error)
            o1_new_bias = o1.bias + o1_bias_change
            o1.updateBias(o1_new_bias)


# This method is responsible for making predictions from our model for our testing set. It works very similarly to our
# training method except it doesn't do any back propagation to update any weights or biases. When it calculates our
# output of our output layer it rounds its prediction accordingly to one of our binary options (0 or 1). Lastly, we
# return these predictions.
def predict():
    m, n = x_test.shape
    predictions = []
    for i in range(m):
        current_sample = x_test[i]
        h1.calcNetInput(h1.weights, current_sample, h1.bias)
        h2.calcNetInput(h2.weights, current_sample, h2.bias)
        h3.calcNetInput(h3.weights, current_sample, h3.bias)
        h4.calcNetInput(h4.weights, current_sample, h4.bias)
        h1.calcOutput(h1.net_input)
        h2.calcOutput(h2.net_input)
        h3.calcOutput(h3.net_input)
        h4.calcOutput(h4.net_input)
        o_input = (h1.output * o1.weights[0]) + (h2.output * o1.weights[1]) + (h3.output * o1.weights[2]) + (
                h4.output * o1.weights[3]) + o1.bias
        o1.updateInput(o_input)
        o1.calcOutput(o1.net_input)
        if o1.output >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# This is our class accuracy method. It is responsible for determining our classification accuracy and returning it.
# It works by getting the length of our predicted set and adding a 1 to a counter variable everytime it finds a correct
# prediction. Lastly, our counter variable divided by our length.
def class_accuracy(actual, predicted):
    m = len(predicted)
    count = 0
    for p in range(m):
        if predicted[p] == actual[p]:
            count = count + 1
    return count / m


# This is our main method. It first loads in our data set and prepares it. Then it initializes and constructs all our
# network's nodes. Next we call our trainModel method followed by our predict method. Lastly, we call our class_accuracy
# method and print it's result.
data = load_breast_cancer()
list(data.target_names)
['malignant', 'benign']
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
w1, b1 = initialize_Node(30)
h1 = Node(w1, b1, 0, 0, 0)
w2, b2 = initialize_Node(30)
h2 = Node(w2, b2, 0, 0, 0)
w3, b3 = initialize_Node(30)
h3 = Node(w3, b3, 0, 0, 0)
w4, b4 = initialize_Node(30)
h4 = Node(w4, b4, 0, 0, 0)
w5, b5 = initialize_Node(4)
o1 = Node(w5, b5, 0, 0, 0)
trainModel(1000)
f_predictions = predict()
score2 = class_accuracy(y_test, f_predictions)
print()
print("Classification Accuracy:")
print(score2)
