# Deep-Neural-Network
A from-scratch implementation of a deep artificial neural network that supports any number of hidden layers / neurons per layer (not optimized for speed, made just for fun).

By default, this model works to identify handwritten digits from the MNIST handwritten character dataset.

The number of hidden layers can be specified, as well as the number of neurons, controllable on a per-layer basis.

Example usage: ```py
# instantiates a network with 784 input neurons, two hidden layers of 16 neurons each, with 10 output neurons (sigmoid activation by default)
nn = NeuralNetwork(784, [16,16], 10) 

# train the network for 10 iterations
nn.train(training_inputs, training_outputs, 10)

# pass some data through the network
result = nn.feed_forward(test_image.flatten().reshape(784).tolist())
```
