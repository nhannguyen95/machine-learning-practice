import numpy as np
import scipy.io as sio

# The data is from Coursera Machine Learning course/week4/ex3data1.mat.
# There are 5000 training examples, each is a 20x20 pixels grayscale
# image of digit 0-9.
data = sio.loadmat('../data/data5.mat')

X = data['X']  # The 20x20 grid of pixels is unrolled into a 400-dimensional vector.
               # Thus X.shape = (5000, 400).
y = data['y']  # The label for 5000 training examples.
               # Note: digit '0' is labeled as '10'.

# Some useful variables.
m, n = X.shape
num_labels = len(np.unique(y))  # Number of labels.

# Load pre-trained neural network coefficient matrices.
coef = sio.loadmat('../data/nn_coef.mat')

# Since we have 1 hidden layer, there are 2 coefficient matrices
# coressponding to input layer and that hidden layer.
# Hidden layer has 25 hidden units.
theta1 = coef['Theta1']  # 25 x 401
theta2 = coef['Theta2']  # 10 x 26

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(theta1, theta2, X):
    # Useful variables.
    m = X.shape[0]

    # Add bias term
    X = np.column_stack((np.ones((m, 1)), X))

    # Activation values of hidden layer
    z2 = X.dot(theta1.T)  # 5000 x 25
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))  # Bias term

    # Activation values of output layer
    z3 = a2.dot(theta2.T)  # 5000 x 10
    a3 = sigmoid(z3)

    # Pick the class that has the most propability of each example
    return (np.argmax(a3, axis = 1) + 1).reshape((m, 1))

# Perform the feedforward propagation to predict y
pred = predict(theta1, theta2, X)

# Accuracy
print('Accuracy: ', np.mean((pred == y).astype(float)) * 100)
