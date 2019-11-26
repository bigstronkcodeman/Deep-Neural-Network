# Deep Neural Network implementation (Multi-layer Perceptron)
# Author: Phillip Boudreau
# Date: 11/26/2019

import numpy as nump

class NeuralNet:
    def __init__(self, inputs, hidden_neurons, outputs):
        """
        Initialize the random synaptic weights of the neural network. 
        Args: 
        inputs: number of inputs on the input layer of the network
        hidden_neurons: a list of numbers [x1,x2,...,xn] where xi specifies the number of neurons
                        in hidden layer i
        outputs: size of output vector
        """
        #seed for reproducibility
        nump.random.seed(23)

        # initialize/store random synaptic weights for the first layer (i.e. input layer) of the network
        weight_matrix = nump.random.uniform(-1, 1, (hidden_neurons[0], inputs + 1))
        self.synaptic_weights = [weight_matrix]

        # initialize/store random synaptic weights for the hidden layers of the network
        for i in range(1, len(hidden_neurons)):
            weight_matrix = nump.random.uniform(-1, 1, (hidden_neurons[i], hidden_neurons[i - 1] + 1))
            self.synaptic_weights.append(weight_matrix)

        # initialize/store random synaptic weights for the last layer (i.e. output layer) of the network
        weight_matrix = nump.random.uniform(-1, 1, (outputs, hidden_neurons[-1] + 1))
        self.synaptic_weights.append(weight_matrix)


    # sigmoid normalization function - used for neuron activation
    def sigmoid(self, x):
        return 1 / (1 + nump.exp(-x))


    # sigmoid derivative - used for synaptic weight / bias tuning during backpropagation
    def sigmoid_derivative(self, x):
        e_neg_x = nump.exp(-x)
        return e_neg_x / ((1 + e_neg_x) * (1 + e_neg_x))


    def forward_feed(self, inputs):
        """
        Propagate input signals forward through layers of the network.
        Returns the outputs from the final layer of the network.
        Args:
        inputs: a 1-dimensional input (column) vector
        """
        # make copy of inputs and append a 1 for bias
        input_layer = inputs
        input_layer.append(1)

        # propagate through input layer of network
        new_input = nump.append(self.sigmoid(nump.matmul(self.synaptic_weights[0], input_layer)), [1])

        # propagate through hidden layers of network
        for i in range(1, len(self.synaptic_weights) - 1):
            new_input = nump.append(self.sigmoid(nump.matmul(self.synaptic_weights[i], new_input)), [1])

        # propagate through final layer of network and return the result
        output = self.sigmoid(nump.matmul(self.synaptic_weights[-1], new_input))
        return output


nn = NeuralNet(10, [100]*100, 10)
print(nn.forward_feed([0,1,2,3,4,5,6,7,8,9]))
        