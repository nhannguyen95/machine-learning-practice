import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio
import random

# The data is from Coursera Machine Learning course/week4/ex3data1.mat
# There are 5000 training examples, each is a 20x20 pixels grayscale
# image of digit 0-9.
data = sio.loadmat('../data/data5.mat')

X = data['X']  # The 20x20 grid of pixels is unrolled into a 400-dimensional vector.
               # Thus X.shape = (5000, 400).
y = data['y']  # The label for 5000 training examples.
               # Note: digit '0' is labeled as '10'

# Some useful variables
m, n = X.shape
num_labels = len(np.unique(y))  # Number of labels

# Randomly select 100 data points to display
sel = X[random.sample(range(0, m), 100), :]

def display_data(X):
    # Specify figsize to avoid unexpected gap between cells in grid
    fig = plt.figure(figsize = (6, 6))

    # Compute size of grid
    m, n = X.shape
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    grid = gridspec.GridSpec(display_rows, display_cols,
                             wspace = 0.0, hspace = 0.0)

    for i in range(m):
        ax = plt.Subplot(fig, grid[i])
        img_size = int(np.sqrt(n))
        ax.imshow(X[i, :].reshape((img_size, img_size)).T, cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

print('Displaying data...')
display_data(sel)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_gradient_descent(X, y, theta_init, alpha, lamda, max_iter = 400):
    m, n = X.shape
    iter = 0
    theta = theta_init
    while iter < max_iter:
        # Batch Gradient Descent
        error = sigmoid(X.dot(theta)) - y
        theta = theta - alpha * (1 / m) * X.T.dot(error) - lamda * (1 / (n - 1)) * theta
        theta[0] += lamda * (1 / (n - 1)) * theta[0]

        iter += 1
    return theta

def oneVsAll(X, y, num_labels, alpha, lamda):
    m = X.shape[0]
    X = np.column_stack((np.ones((m, 1)), X))  # Add bias term
    all_theta = np.zeros((num_labels, X.shape[1]))
    for i in range(1, num_labels + 1):
        theta_init = np.zeros((X.shape[1], 1))
        theta = logistic_gradient_descent(X, (y == i).astype(float), theta_init, alpha, lamda)

        all_theta[i - 1, :] = theta.T

    return all_theta

print('Training One-vs-All Logistic Regression...')
print('(This may take a while)')
alpha = 0.05  # Learning rate
lamda = 1     # Regularization term lambda
all_theta = oneVsAll(X, y, num_labels, alpha, lamda)

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    X = np.column_stack((np.ones((m, 1)), X))  # Add bias term
    prob = sigmoid(X.dot(all_theta.T))
    return np.argmax(prob, axis = 1) + 1

# Accuracy
y_pred = predictOneVsAll(all_theta, X).reshape(m, 1)
print('Accuracy: ', np.mean((y_pred == y).astype(float)) * 100)
