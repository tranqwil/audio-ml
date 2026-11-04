import numpy as np
import math

class MLP:

    def __init__(self, num_inputs=3, num_hidden = [3, 5], num_outputs = 2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # a list of the layers, with each index indicating the number of elements in that layer e.g. [3, 3, 5, 2]
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        self.weights = []

        # Creates a matrix w with layers[i] rows and layers[i+1] columns. The row is the neuron the connection comes from, and the column is the neuron it goes to. 
        # For example:
        #                  to neuron 1   to neuron 2   to neuron 3
        # from neuron 1  [   w1,1          w1,2          w1,3   ]
        # from neuron 2  [   w2,1          w2,2          w2,3   ]
        # from neuron 3  [   w3,1          w3,2          w3,3   ]
        #
        # This is repeated for len(layers) -1 until self.weights is a list of matrixes, where each matrix in the list is a matrix of neuron to neuron weights as described above

        for i in range (len(layers) - 1): #iterate through layers
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

    def forward_propogate(self, inputs):
        activations = inputs # before we have transformed any data, activations is just the incoming data

        for w in self.weights:
            # calculate the net inputs
            # perform a matrix multiplication of the inputs (either the input data, or the activation from the previous layer), with the weights between a given layer.

            net_inputs = np.dot(activations, w)

            # calculate the activations

            # pass the net inputs to the sigmoid function and store the result in the activations list.
            # this will be repeated for every matrix of weights. Once this for loop exits, we have processed the activations for the final layer, the output layer, at which point we return
            activations = self._sigmoid(net_inputs)

        return activations

if __name__ == "__main__":
    # instantiate our MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward prop
    outputs = mlp.forward_propogate(inputs)

    # output the result
    print("Network Input is: {}".format(inputs))
    print("Network Output is: {}".format(outputs))



        

    