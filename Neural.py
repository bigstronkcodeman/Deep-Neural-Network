# Deep Neural Network implementation (Multi-layer Perceptron)
# Author: Phillip Boudreau
# Date: 09/18/2020
import numpy as np
import random as rand
from emnist import extract_training_samples, extract_test_samples

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

epochs = []
costs = []

class NeuralNetwork:
    def __init__(self, num_inputs, hidden_layers, num_outputs, activation_function = "sigmoid", layer_activations = []):
        """
        Initializes the random synaptic weights and biases of the neural network.
        Args:
        num_inputs: an integer denoting the number of inputs to the network
        hidden_layers: a 1-dimensional array [x1,x2,...,xn] where xi represents the number of 
                       hidden neurons on layer i
        num_outputs: number of neurons on the output layer
        activation_function: specifies an activation function to use for the entire network
                             if the user does not wish to use different activation functions
                             for different layers
        layer_activations: a list of strings denoting the activation functions [af1,af2,...,afn] where
                           afi represents the activation function to use for layer i
        """
        # range for weights and bias initialization
        a = -1
        b = 1

        # initialize learning rate
        self.learning_rate = 0.1

        # initialize random synaptic weights and biases for first layer (input
        # layer) of the network
        self.synaptic_weights = [np.random.uniform(a, b, (hidden_layers[0], num_inputs))]
        self.biases = [np.random.uniform(a, b, (hidden_layers[0], 1))]

        # initialize random synaptic weights and biases for hidden layers of
        # the network
        for i in range(1, len(hidden_layers)):
            self.synaptic_weights.append(np.random.uniform(a, b, (hidden_layers[i], hidden_layers[i - 1])))
            self.biases.append(np.random.uniform(a, b, (hidden_layers[i], 1)))

        # initialize random synaptic weights and biases for last layer (output
        # layer) of the network
        self.synaptic_weights.append(np.random.uniform(a, b, (num_outputs, hidden_layers[-1])))
        self.biases.append(np.random.uniform(a, b, (num_outputs, 1)))

        # initialize main activation function (to use if layer-specific activations are not specified)
        self.activation_function = self.sigmoid
        self.activation_function_derivative = self.sigmoid_derivative
        if activation_function == "relu":
            self.activation_function = self.relu
            self.activation_function_derivative = self.relu_derivative
        if activation_function == "leaky-relu":
            self.activation_function = self.leaky_relu
            self.activation_function_derivative = self.leaky_relu_derivative

        # initialize array of activation functions where the index of the function corresponds to which
        # layer it is being used to activate
        self.activation_functions = []
        self.activation_functions_derivatives = []
        if len(layer_activations) > 0 and len(layer_activations) == len(hidden_layers) + 1:
           for i in range(len(layer_activations)):
                if layer_activations[i] == "leaky-relu":
                    self.activation_functions.append(self.leaky_relu)
                    self.activation_functions_derivatives.append(self.leaky_relu_derivative)
                elif layer_activations[i] == "relu":
                    self.activation_functions.append(self.relu)
                    self.activation_functions_derivatives.append(self.relu_derivative)
                else:
                    self.activation_functions.append(self.sigmoid)
                    self.activation_functions_derivatives.append(self.sigmoid_derivative)
        else:
            for i in range(len(hidden_layers) + 1):
                self.activation_functions.append(self.activation_function)
                self.activation_functions_derivatives.append(self.activation_function_derivative)

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derivative(self, x):
        if not np.isscalar(x):
            return np.array([0 if xi <= 0 else 1.0 for xi in x])
        else:
            return 0 if x <= 0 else 1.0

    def leaky_relu(self, x):
        if not np.isscalar(x):
            return np.array([0.1 * xi if xi <= 0 else xi for xi in x])
        else:
            return 0.1 * x if x <= 0 else x

    def leaky_relu_derivative(self, x):
        if not np.isscalar(x):
            return np.array([0.1 if xi <= 0 else 1.0 for xi in x])
        else:
            return 0.1 if x <= 0 else 1.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # feed inputs through the network
    def feed_forward(self, input):
        new_input = self.activation_functions[0](np.matmul(self.synaptic_weights[0], np.array([input]).T) + self.biases[0])

        for i in range(1, len(self.synaptic_weights) - 1):
            new_input = self.activation_functions[i](np.matmul(self.synaptic_weights[i], new_input) + self.biases[i])

        return self.activation_functions[-1](np.matmul(self.synaptic_weights[-1], new_input) + self.biases[-1])

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
            
        global epochs
        epochs = [i for i in range(iterations)]

        for iters in range(iterations):
            # calculate total cost of network
            #average_output_error = np.full((len(training_outputs[0]), 1), 0.0)
            average_output_error = 0
            for inp, outp in zip(training_inputs, training_outputs):
                training_output = np.array([outp]).T
                diff = np.array(training_output - self.feed_forward(inp))
                average_output_error += np.sum(np.array(diff * diff))
            average_output_error /= len(training_outputs)
            average_output_error = np.sum(average_output_error)
            costs.append(average_output_error.item())
            
            print('Training iteration ', iters, ' total cost: ', average_output_error)
            if average_output_error <= 0.01:
                break

            for inp, outp in zip(training_inputs, training_outputs):
                training_input = np.array([inp]).T
                training_output = np.array([outp]).T

                # pump inputs through first layer of network and save result
                layer_results = [self.activation_functions[0](np.matmul(self.synaptic_weights[0], training_input) + self.biases[0])]

                # sequentially pump inputs through hidden layers of the network and save results
                for i in range(1, len(self.synaptic_weights) - 1):
                    layer_results.append(self.activation_functions[i](np.matmul(self.synaptic_weights[i], layer_results[-1]) + self.biases[i]))

                # pump inputs through final layer of network and save result
                layer_results.append(self.activation_functions[-1](np.matmul(self.synaptic_weights[-1], layer_results[-1]) + self.biases[-1]))

                # calculate output error and deltas for synaptic weights and biases on last layer of network
                act_func_index = -1
                output_error = np.array(layer_results[-1] - training_output)
                output_weight_deltas = np.full(self.synaptic_weights[-1].shape, 1.0)
                output_bias_deltas = np.full(self.biases[-1].shape, 1.0)
                for i in range(len(output_weight_deltas)):
                    for j in range(len(output_weight_deltas[i])):
                        output_weight_deltas[i][j] *= output_error[i][0]                                                             \
                                                    * self.activation_functions_derivatives[act_func_index](layer_results[-1][i][0]) \
                                                    * layer_results[-2][j][0]                                                        \
                                                    * self.learning_rate
                    output_bias_deltas[i][0] *= output_error[i][0]                                                             \
                                              * self.activation_functions_derivatives[act_func_index](layer_results[-1][i][0]) \
                                              * self.learning_rate
                act_func_index -= 1
                weight_deltas = [output_weight_deltas]
                bias_deltas = [output_bias_deltas]

                # calculate synaptic weight and bias deltas for hidden layers
                save = [output_error[i][0] for i in range(len(output_error))]
                for layer in range(len(self.synaptic_weights) - 2, 0, -1):
                    next_save = []
                    layer_weight_deltas = np.full(self.synaptic_weights[layer].shape, 1.0)
                    layer_bias_deltas = np.full(self.biases[layer].shape, 1.0)
                    for i in range(len(layer_weight_deltas)):
                        propagated_error = 0
                        for k in range(len(self.synaptic_weights[layer + 1])):
                            error = self.synaptic_weights[layer + 1][k][i]
                            error *= self.activation_functions_derivatives[act_func_index + 1](layer_results[layer + 1][k][0]) \
                                   * save[k]
                            propagated_error += error
                        next_save.append(propagated_error)
                        for j in range(len(layer_weight_deltas[i])):
                            layer_weight_deltas[i][j] *= layer_results[layer - 1][j][0] \
                                                       * self.activation_functions_derivatives[act_func_index](layer_results[layer][i][0]) \
                                                       * propagated_error                                                                  \
                                                       * self.learning_rate
                        layer_bias_deltas[i][0] *= self.activation_functions_derivatives[act_func_index](layer_results[layer][i][0]) \
                                                 * propagated_error \
                                                 * self.learning_rate
                    act_func_index -= 1
                    save = next_save
                    weight_deltas.insert(0, layer_weight_deltas)
                    bias_deltas.insert(0, layer_bias_deltas)

                # calculate synaptic weight and bias deltas for input layer
                input_weight_deltas = np.full(self.synaptic_weights[0].shape, 1.0)
                input_bias_deltas = np.full(self.biases[0].shape, 1.0)
                for i in range(len(input_weight_deltas)):
                    propagated_error = 0
                    for k in range(len(self.synaptic_weights[1])):
                        error = self.synaptic_weights[1][k][i]
                        error *= self.activation_functions_derivatives[act_func_index + 1](layer_results[1][k][0]) \
                               * save[k]
                        propagated_error += error
                    for j in range(len(input_weight_deltas[i])):
                        input_weight_deltas[i][j] *= training_input[j][0]                                                          \
                                                   * self.activation_functions_derivatives[act_func_index](layer_results[0][i][0]) \
                                                   * propagated_error                                                              \
                                                   * self.learning_rate
                    input_bias_deltas[i][0] *= self.activation_functions_derivatives[act_func_index](layer_results[0][i][0]) \
                                             * propagated_error                                                              \
                                             * self.learning_rate                                                            
                weight_deltas.insert(0, input_weight_deltas)
                bias_deltas.insert(0, input_bias_deltas)

                # update synaptic weights and biases
                for i in range(len(weight_deltas)):
                    self.synaptic_weights[i] -= weight_deltas[i]
                    self.biases[i] -= bias_deltas[i]
                    
                    

def frange(x, y, step):
    while x < y:
        yield x
        x += step

# for graphing 3-d functions "learned" by the network
def graph_3D():
    x = []
    y = []
    z = []
    for i in frange(0, 1.0, 0.025):
        for j in frange(0, 1.0, 0.025):
            ff = nn.feed_forward([i,j])
            x.append(i)
            y.append(j)
            z.append(ff)
    plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, c='b', marker='.')
    plt.show()

# plot cost of network over training
def plot_cost():
    plt.plot(epochs, costs, color='red', marker='.')
    plt.title('Network Training Cost vs. Training Iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()

def main():
    np.set_printoptions(suppress=True)

    # prepare training and testing datasets
    training_images, training_labels = extract_training_samples('digits')
    test_images, test_labels = extract_test_samples('digits')
    training_images = training_images[0:10000]
    training_labels = training_labels[0:10000]
    tr_i = [training_images[i].flatten().reshape(784).tolist() for i in range(len(training_images))]
    for i in range(len(tr_i)):
        for j in range(len(tr_i[i])):
            tr_i[i][j] /= 255.0
    tr_o = [[x] for x in training_labels.tolist()]
    tr_o = [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01] for i in range(len(training_labels))]
    for i in range(len(tr_o)):
        tr_o[i][training_labels[i]] = 0.99
       
    # initialize and train the network
    nn = NeuralNetwork(784, [16,16], 10)
    nn.train(tr_i, tr_o, 1000)

    # gauge performance
    correct = 0
    for test_image, test_label in zip(test_images[0:500], test_labels[0:500]):
        result = nn.feed_forward(test_image.flatten().reshape(784).tolist())
        print("network result:\n", result);
        max = 0
        guess = -1
        for i, res in enumerate(result):
            if res > max:
                max = res
                guess = i
        print('network thinks this is a: ', guess)
        print("real answer:", test_label)
        if guess == int(test_label):
            correct += 1
    print('network was correct on ', correct, '/', 500, 'images')

if __name__ == '__main__':
    main()
