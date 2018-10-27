import numpy as np

# Definitions of different activation functions
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z
def softplus(X):
    z = np.log(1 + np.exp(X))
    return z
def ReLU(X):
    z = abs(X) * (X > 0)
    return z

#general activation function for choosing different activation functions
def act(X, activation):
    if activation == "sigmoid":
        return sigmoid(X)
    if activation == "tanh":
        return np.tanh(X)
    if activation == "ReLU":
        return ReLU(X)
    if activation == "softplus":
        return softplus(X)
    raise ValueError("Not a proper activation function")

#derivatives of the different activation functions
def Gprime(Z, A, activation):
    if activation == "sigmoid":
        return (np.exp(-Z) * np.power(A, 2))
    if activation == "tanh":
        return (1 - np.power(A, 2))
    if activation == "softplus":
        return sigmoid(Z)
    if activation == "ReLU":
        return 1.0 * (Z > 0)

# initialize the weights and biases
def initialize(n_x, n_h, n_h1, n_y):
    
    # fills the weight matricies with random numbers (0, 1) and fill the bias matricies with zeros
    #W1 = np.random.randn(n_h, n_x)
    W1 = np.random.normal(0.0, pow(n_h, -0.5), (n_h, n_x))
    b1 = np.zeros((n_h, 1))
    #W2 = np.random.randn(n_h1, n_h)
    W2 = np.random.normal(0.0, pow(n_h1, -0.5), (n_h1, n_h))
    b2 = np.zeros((n_h1, 1))
    #W3 = np.random.randn(n_y, n_h1)
    W3 = np.random.normal(0.0, pow(n_y, -0.5), (n_y, n_h1))
    b3 = np.zeros((n_y, 1))

    # dictionary of w&b to stop from having to list them as parameters
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2,
        "W3" : W3,
        "b3" : b3
        }
    
    return parameters

def options(learning_rate, activation, activation_out):
    
    options = {
        "act" : activation,
        "actout" : activation_out,
        "lr" : learning_rate
        }

    return options


# trains the neural network
def train(X, Y, parameters, options):

    #import user options
    actin = options["act"]
    actout = options["actout"]
    lr = options["lr"]
    
    # import the weights and biases
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # compute the activations 
    Z1 = np.dot(W1, X) + b1
    A1 = act(Z1, actin)
    Z2 = np.dot(W2, A1) + b2
    A2 = act(Z2, actin)
    Z3 = np.dot(W3, A2) + b3
    A3 = act(Z3, actout)

    # compute the derivatives for gradient descent 
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis = 1, keepdims = True)
    dZ2 = np.dot(W3.T, dZ3) * (Gprime(Z2, A2, actin))
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (Gprime(Z1, A1, actin)) # the second term is d/dZ1(sigmoid(Z1)) and will need to be altered if the act. function is changed
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis = 1, keepdims = True)

    # updating the parameters with gradient descent 
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W3 = W3 - lr * dW3
    b3 = b3 - lr * db3

    # updated parameters dictionary
    parameters = {
        "W1" : W1,
        "W2" : W2,
        "W3" : W3,
        "b1" : b1,
        "b2" : b2,
        "b3" : b3
        }

    return parameters, A3

# function for testing the trained neural network
def test(X, parameters, options):

    #import user options
    actin = options["act"]
    actout = options["actout"]
    lr = options["lr"]

    #imporing the trained weights and biases
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
                    
    # computing the activations with the updated weights and biases
    Z1 = np.dot(W1, X) + b1
    A1 = act(Z1, actin)
    Z2 = np.dot(W2, A1) + b2
    A2 = act(Z2, actin) 
    Z3 = np.dot(W3, A2) + b3
    A3 = act(Z3, actout)

    return A3
