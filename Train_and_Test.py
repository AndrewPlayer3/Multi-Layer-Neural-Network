import numpy as np
import Multi_Layer_NN as nn
import time

#initializing the # of input nodes and # of output nodes
n_x = 784
n_y = 10

# promts user for their desired settings
print("===Settings===")
while True:
    try:
        learning_rate = float(input("Learning rate: "))
        break
    except ValueError:
        print("Not a valid float. Try again...")
while True:        
    try:
        n_h = int(input("Number of hidden nodes in hl_0: "))
        break
    except ValueError:
        print("Not a valid int. Try again...")
while True:        
    try:
        n_h1 = int(input("Number of hidden nodes in hl_1: "))
        break
    except ValueError:
        print("Not a valid int. Try again...")   
activation = input("Choose hidden layer activation function: ")
activation_out = input("Choose output layer activation function: ")
while True:        
    try:
        epochs = int(input("Number of epochs: "))
        break
    except ValueError:
        print("Not a valid int. Try again...")  

#initializing the user options
options = nn.options(learning_rate, activation, activation_out)

#initializing weights and biases and outputing dictionary containing them
parameters = nn.initialize(n_x, n_h, n_h1, n_y)

#getting training data
training_data_file = open("mnist_train.csv")
training_data = training_data_file.readlines()
training_data_file.close()

#training the neural network
print("\n===Training===")
start = time.time()
n = 0
epoch_num = 1
for i in range(epochs):
    for digit in training_data:        
        n += 1
        img_array = digit.split(',')
        X = (np.asfarray(img_array[1:]) / 255.0 * 0.99) + 0.01
        X = X.reshape((784,1))
        Y = np.zeros((n_y, 1)) + 0.01
        Y[int(img_array[0])] = 0.99
        parameters, A3 = nn.train(X, Y, parameters, options)
        if((n == 1) and (epoch_num == 1)):
            print("Epoch: 1")
        if(n % 10000 == 0):
            print(n,"Trained...")
    n = 0
    epoch_num += 1
    if epoch_num <= epochs:
        print("\nEpoch:",epoch_num)
end = time.time()
print("\nTraining complete.")
print("Time elapsed:",end - start,"\bs")

#getting testing data
test_data_file = open("mnist_test.csv")
testing_data = test_data_file.readlines()
test_data_file.close()

#testing
print("\n===Testing===")
number_correct = 0
for digit in testing_data:
    img_array = digit.split(',')
    X = (np.asfarray(img_array[1:]) / 255.0 * 0.99) + 0.01
    X = X.reshape((784, 1)) 
    guess = nn.test(X, parameters, options)
    if np.argmax(guess) == int(img_array[0]):
        number_correct += 1

# calculates and prints the accuracy of the neural net over the test set 
accuracy = number_correct / 10000.0
print("Accuracy:", accuracy*100,"\b%\n")