import math

def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y

def activate(inputs, weights):
    # get weighted sum
    weightedSum = 0
    for x, w in zip(inputs, weights):
        
        weightedSum += x * w

    #perform sigmoid function
    return sigmoid(weightedSum)


    

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]

    output = activate(inputs, weights)
    print(output)





