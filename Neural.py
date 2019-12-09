# Deep Neural Network implementation (Multi-layer Perceptron)
# Author: Phillip Boudreau
# Date: 11/26/2019
import numpy as np
import random as rand

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def sum(li):
    s = 0
    for i in range(len(li)):
        s += li[i]
    return s

class NeuralNetwork:
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        """
        Initializes the random synaptic weights and biases of the neural network.
        Args:
        num_inputs: an integer denoting the number of inputs to the network
        hidden_layers: a 1-dimensional array [x1,x2,...,xn] where xi represents the number of 
                       hidden neurons on layer i
        num_outputs: number of neurons on the output layer
        """
        # initialize learning rate
        self.learning_rate = 1

        # initialize random synaptic weights and biases for first layer (input
        # layer) of the network
        self.synaptic_weights = [np.random.uniform(-1, 1, (hidden_layers[0], num_inputs))]
        self.biases = [np.random.uniform(-1, 1, (hidden_layers[0], 1))]

        # initialize random synaptic weights and biases for hidden layers of
        # the network
        for i in range(1, len(hidden_layers)):
            self.synaptic_weights.append(np.random.uniform(-1, 1, (hidden_layers[i], hidden_layers[i - 1])))
            self.biases.append(np.random.uniform(-1, 1, (hidden_layers[i], 1)))

        # initialize random synaptic weights and biases for last layer (output
        # layer) of the network
        self.synaptic_weights.append(np.random.uniform(-1, 1, (num_outputs, hidden_layers[-1])))
        self.biases.append(np.random.uniform(-1, 1, (num_outputs, 1)))

    # sigmoid logistic function, used for neuronal activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative, used for backpropagation
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, input):
        new_input = self.sigmoid(np.matmul(self.synaptic_weights[0], np.array([input]).T) + self.biases[0])

        for i in range(1, len(self.synaptic_weights) - 1):
            new_input = self.sigmoid(np.matmul(self.synaptic_weights[i], new_input) + self.biases[i])

        return self.sigmoid(np.matmul(self.synaptic_weights[-1], new_input) + self.biases[-1])

    def train(self, training_inputs, training_outputs, iterations):
        """
        Uses backpropagation (gradient descent) to adjust weights and biases
        so as to correctly identify classes in the given data set
        Args:
        training_inputs: 1-dimensional list of input vectors
        training_output: 1-dimensional list of output vectors
        iterations: number of training epochs
        """
        if len(training_inputs) != len(training_outputs):
            raise Exception('Different number of training inputs and training outputs!')
            
        for iters in range(iterations):
            average_output_error = np.full((len(training_outputs[0]), 1), 0.0)
            for inp, outp in zip(training_inputs, training_outputs):
                training_output = np.array([outp]).T
                diff = np.array(training_output - self.feed_forward(inp))
                average_output_error += np.array(diff * diff)
            average_output_error /= len(training_outputs)
            print('Training iteration ', iters, ' total error: ', average_output_error)
            for inp, outp in zip(training_inputs, training_outputs):
                training_input = np.array([inp]).T
                training_output = np.array([outp]).T

                # pump inputs through first layer of network
                layer_results = [self.sigmoid(np.matmul(self.synaptic_weights[0], training_input) + self.biases[0])]

                # sequentially pump inputs through hidden layers of the network
                for i in range(1, len(self.synaptic_weights) - 1):
                    layer_results.append(self.sigmoid(np.matmul(self.synaptic_weights[i], layer_results[-1]) + self.biases[i]))

                # pump inputs through final layer of network
                layer_results.append(self.sigmoid(np.matmul(self.synaptic_weights[-1], layer_results[-1]) + self.biases[-1]))

                # calculate output error and deltas for weights on last layer of network
                output_error = np.array(layer_results[-1] - training_output)
                output_weight_deltas = np.full(self.synaptic_weights[-1].shape, 1.0)
                output_bias_deltas = np.full(self.biases[-1].shape, 1.0)
                for i in range(len(output_weight_deltas)):
                    for j in range(len(output_weight_deltas[i])):
                        output_weight_deltas[i][j] *= output_error[i][0]
                        output_weight_deltas[i][j] *= self.sigmoid_derivative(layer_results[-1][i][0])
                        output_weight_deltas[i][j] *= layer_results[-2][j][0]
                        output_weight_deltas[i][j] *= self.learning_rate
                    output_bias_deltas[i][0] *= output_error[i][0]
                    output_bias_deltas[i][0] *= self.sigmoid_derivative(layer_results[-1][i][0])
                    output_bias_deltas[i][0] *= self.learning_rate
                weight_deltas = [output_weight_deltas]
                bias_deltas = [output_bias_deltas]

                # calculate weight and bias deltas for hidden layers
                save = [output_error[i][0] for i in range(len(output_error))]
                for layer in range(len(self.synaptic_weights) - 2, 0, -1):
                    next_save = []
                    layer_weight_deltas = np.full(self.synaptic_weights[layer].shape, 1.0)
                    layer_bias_deltas = np.full(self.biases[layer].shape, 1.0)
                    for i in range(len(layer_weight_deltas)):
                        propagated_error = 0
                        for k in range(len(self.synaptic_weights[layer + 1])):
                            error = self.synaptic_weights[layer + 1][k][i]
                            error *= self.sigmoid_derivative(layer_results[layer + 1][k][0])
                            error *= save[k]
                            propagated_error += error
                        next_save.append(propagated_error)
                        for j in range(len(layer_weight_deltas[i])):
                            layer_weight_deltas[i][j] *= layer_results[layer - 1][j][0]
                            layer_weight_deltas[i][j] *= self.sigmoid_derivative(layer_results[layer][i][0])
                            layer_weight_deltas[i][j] *= propagated_error
                            layer_weight_deltas[i][j] *= self.learning_rate
                        layer_bias_deltas[i][0] *= self.sigmoid_derivative(layer_results[layer][i][0])
                        layer_bias_deltas[i][0] *= propagated_error
                        layer_bias_deltas[i][0] *= self.learning_rate
                    save = next_save
                    weight_deltas.insert(0, layer_weight_deltas)
                    bias_deltas.insert(0, layer_bias_deltas)

                # calculate weight and bias deltas for input layer
                input_weight_deltas = np.full(self.synaptic_weights[0].shape, 1.0)
                input_bias_deltas = np.full(self.biases[0].shape, 1.0)
                for i in range(len(input_weight_deltas)):
                    propagated_error = 0
                    for k in range(len(self.synaptic_weights[1])):
                        error = self.synaptic_weights[1][k][i]
                        error *= self.sigmoid_derivative(layer_results[1][k][0])
                        error *= save[k]
                        propagated_error += error
                    for j in range(len(input_weight_deltas[i])):
                        input_weight_deltas[i][j] *= training_input[j][0]
                        input_weight_deltas[i][j] *= self.sigmoid_derivative(layer_results[0][i][0])
                        input_weight_deltas[i][j] *= propagated_error
                        input_weight_deltas[i][j] *= self.learning_rate
                    input_bias_deltas[i][0] *= self.sigmoid_derivative(layer_results[0][i][0])
                    input_bias_deltas[i][0] *= propagated_error
                    input_bias_deltas[i][0] *= self.learning_rate
                weight_deltas.insert(0, input_weight_deltas)
                bias_deltas.insert(0, input_bias_deltas)

                for i in range(len(weight_deltas)):
                    self.synaptic_weights[i] -= weight_deltas[i]
                    self.biases[i] -= bias_deltas[i]


def frange(x, y, step):
    while x < y:
        yield x
        x += step

def bitz(num, n):
    binary = []
    while num != 0:
        bit = num % 2
        binary.insert(0, bit)
        num = int(num / 2)
    while len(binary) < n:
        binary.insert(0, 0)
    return binary

#inputs = [[0,0],
#          [0,1],
#          [1,0],
#          [1,1]]
#outputs = [[0],
#           [1],
#           [1],
#           [0]]
nums = [i for i in range(0, 128, 1)]
rand.shuffle(nums)
inputs = [bitz(i,8) for i in nums]
outputs = [[1] if i % 3 == 0 and i > 0 else [0] for i in nums]
nn = NeuralNetwork(8, [25], 1)
nn.train(inputs, outputs, 10000)
for i in range(256):
    result = nn.feed_forward(bitz(i,8))
    if result > 0.5 and i % 3 == 0:
        print(i, ': ', result, ' nn is correct')
    else:
       if result < 0.5 and i % 3 != 0:
           print(i, ': ', result, ' nn is correct')
       else:
           print(i, ': ', result, ' nn is incorrect')

x = []
y = []
for i in range(25):
    x.append(i)
    y.append(nn.feed_forward(bitz(i,8)))
plt.scatter(x,y)
plt.show()
#x = []
#y = []
#z = []
#for i in frange(0, 1.0, 0.0333):
#    for j in frange(0, 1.0, 0.0333):
#        ff = nn.feed_forward([i,j])
#        x.append(i)
#        y.append(j)
#        z.append(ff)
#plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, c='g', marker='o')
#plt.show()
