import numpy as np
import tensorflow as tf
from random import random
from sklearn.model_selection import train_test_split

#TODO
# build the model []
# compile the model []
# train model []
# evaluate the model []
# make some predictions []

def generate_dataset(number_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(number_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(10000, 0.3)
    #print("x_test: {}\n".format(x_test))
    #print("y_test: {}\n".format(y_test))

    # build model with 2 -> 5 -> 1 architecture (2 input neurons, 1 hidden layer with 5 neurons and 1 output)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"), # dense layer is a fully connected layer, one that connects all the neurons from the previous layer with the current layer
        tf.keras.layers.Dense(1, activation="sigmoid" )
    ])

    #compile the model
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1) #instantiate an optimizer class with a learning rate of 0.1
    model.compile(optimizer=optimiser, loss="MSE") #omptimizer is stochastic gradient descent, and the loss function is mean squared error

    #train the model
    model.fit(x_train, y_train, epochs=200)

    print("Evaluation: \n")
    #evaluate the model
    model.evaluate(x_test, y_test, verbose=1)

    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print("\nSome preodictions: ")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))
