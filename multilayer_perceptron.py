import numpy as np
from random import random
import math

#TODO
# Save the activations and the derivatives [x]
# Implement backpropogration [x]
# Implement gradient descent [x]
# Implement a 'train' method which uses back propogration and gradient descent [x]
# Train out network with some dummy dataset []
# make some predictions []


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

        activations = [] # somewhere to store our activations. A matrix of values where each row represents a layer and each columns represent an activation for a neuron in a given layer.
        for i in range(len(layers)):
            a = np.zeros(layers[i]) # an array of zeros equal to the number of neurons in each layer - layers[i]
            activations.append(a)
        self.activations = activations

        derivatives = [] # somewhere to store the derivatives for each set of weights where each row represent a set of weights, and each columns represent a weight in a given set
        for i in range(len(layers) - 1): # number of layers - 1 because weights exists between each layer
            d = np.zeros((layers[i], layers[i+1])) # derivatives matrix has the same layout as the weights since derivatives are computed using the weights. and we have 1 derivative for every weight
            derivatives.append(d)
        self.derivatives = derivatives

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

    def forward_propogate(self, inputs):
        activations = inputs # before we have transformed any data, activations is just the incoming data
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate the net inputs
            # perform a matrix multiplication of the inputs (either the input data, or the activation from the previous layer), with the weights between a given layer.

            net_inputs = np.dot(activations, w)

            # calculate the activations

            # pass the net inputs to the sigmoid function and store the result in the activations list.
            # this will be repeated for every matrix of weights. Once this for loop exits, we have processed the activations for the final layer, the output layer, at which point we return
            activations = self._sigmoid(net_inputs)

            # save the activations in the member variable
            self.activations[i + 1] = activations # i + 1 being the layer after the set of weights we are currently iterating on

        return activations

    def backward_propogate(self, error, verbose=False):
        # The objective of backward propagation is to calculate the gradient of the loss function
        # with respect to each set of weights, i.e. how much the loss changes when each weight changes.
        # We need this because the gradient tells us which direction to adjust each weight to reduce the loss.
        # We work backwards through the network because the gradient for one layer depends on the
        # gradients of the layers after it - this is the chain rule. Each layer's gradient is computed
        # from the next layer's, so intermediate results are reused rather than recomputed.

        # Gradient descent - tweaking of the parameters (weights) with respect to the gradient of the errors
        # With gradient descent we take a step to the opposite direction of the gradient
        # the size of that step (learning rate) is a parameter that can be tweaked to obtain better results

        # USEFUL EXPLANATION THAT VELERIO DIDN'T EXPLAIN # 

        # The derivative of the loss E with respect to a weight W is found by multiplying the derivatives
        # of each function the signal passes through between W and E (the chain rule).
        # Where the signal splits across several neurons, the contributions from each route are summed.
        # Why do we do that?  Consider the network below, Changes to weight w impact E through two routes, route 1 via b1 and route 2 via b2
        # Consequently, to calculate the derivative of E with respect to W, we must first calculate to the derivate of E via route 1, and then via route 2; and then sum them
        #           ┌─► b1 ─┐
        #  w ─► a ──┤       ├─► E
        #           └─► b2 ─┘

        # IMPORTANT NOTE ABOUTHE 'error' PARAMETER
        # `error` here is dE/da: the derivative of the loss with respect to the activations of the layer
        # to the right of the weights currently being processed (in the for loop). 
        # On the first iteration it comes straight from the loss function; after that it is the signal passed back from the layer ahead of it.
    

        for i in reversed(range(len(self.derivatives))): # go backwards through each set of derivatives

            # The principle of calculating the derivative of E for a given weight W is that you multiply the derivatives of each function the signal passes through
            # For us, those are the following: the Weighted Sum, the Activation Function and the Loss Function
            # At some point in the future I will attempt to understand better, but for know we can understand that two of those are already given to us:
            # Derivative of the Weighted Sum is the Previous Activations
            # Derivative of the Loss Function is the Error value which is passed to the function (THIS IS BECAUSE THE 'error' value passed to this method is not the error itslef, it is the derivative of the error function )
            # And then we manually calculate the derivative of the Activation Function (self._sigmoid_deriviative())





            activations = self.activations[i+1] # get the activations 'to the right' of these derivatives
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)


            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            # Pass the error signal back one layer: multiplying delta by the transposed weights sums, for each
            # neuron on the left, the contribution of every neuron it feeds (the 'routes' from earlier).
            # The result is dE/da for the previous layer, which becomes `error` for the next iteration (chain rule).
            error = np.dot(delta, self.weights[i].T) 

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error # the error back propogated all the way back to the input layer

        
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def gradient_descent(self, learning_rate):
        # learning rate is the size of the step we want to take against the gradient to optimize from the error function

        for i in range(len(self.weights)):
            weights = self.weights[i] # weights for the current layer
            #print("Original Weights: W{} {}".format(i, weights))
            derivatives = self.derivatives[i] # corresponding derivatives
            weights += derivatives * learning_rate # nudge the weights according to the sign of the derivatives, scaled by the learning rate
            #print("Adjusted Weights: W{} {}".format(i, weights))

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range (epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # perform forward prop
                    output = self.forward_propogate(input)
                
                    #calculate error
                    error = target - output
                
                    # perform backward prop
                    self.backward_propogate(error)
                
                    # apply gradient descent
                    self.gradient_descent(learning_rate)

                    # sum error
                    sum_error += self._mse(target, output)
            print("Error after Epoch {}: {}".format(i, sum_error / len(inputs)))

                
    def _mse(self, target, output):
        # get the average of the squared error
        return np.average((target - output)**2)
        



if __name__ == "__main__":
    # instantiate our MLP
    mlp = MLP(2, [5], 1)

     # create a dataset to train a network for the sum operation
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # train the MLP
    mlp.train(inputs, targets, 1000, 0.1)

    test_inputs = [0.1, 0.2]
    prediction = mlp.forward_propogate(test_inputs)
    print("Prediction: {}".format(prediction))






        

    