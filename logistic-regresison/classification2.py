import numpy as np
import matplotlib.pyplot as plt

# The data is from Coursera Machine Learning course/week3/ex2data2.txt
data = np.loadtxt('../data/data4.txt', delimiter = ',')

# Some useful variables
m = data.shape[0]  # Number of examples
n = data.shape[1]  # Number of comlumns

X = data[:, 0:(n - 1)]          # Score of two different tests on a microchip
Y = data[:, np.newaxis, n - 1]  # The microchip is accepted/rejected (1/0)

print('Mapping features...')
def map_feature(X1, X2, degree):
    X1 = np.atleast_1d(X1)  # Wrap in an array
    X2 = np.atleast_1d(X2)
    X = np.ones((X1.shape[0], 1))  # Includes bias term
    for i in range(1, degree+1):
        for j in range(0, i + 1):
            new_feature = (X1**(i - j)) * (X2**j)  # x1^(i - j) * x2^j
            X = np.column_stack((X, new_feature))
    return X

# Map original features to polynomial features
# in order to fit the data better
degree = 6  # Highest degree of polynomial features
X = map_feature(X[:, np.newaxis, 0], X[:, np.newaxis, 1], degree)

# Feature Normalization using Standardization
mu = np.mean(X[:, 1::], axis = 0)
sd = np.std(X[:, 1::], axis = 0)
X[:, 1::] = (X[:, 1::] - mu) / sd

# Visualize the data
print('Plotting the data...')
plt.figure(1)
neg = np.where(Y == 0)[0]  # index of rejected examples
pos = np.where(Y == 1)[0]  # index of accepted examples
plt.plot(X[neg, 1], X[neg, 2], 'o', color = 'blue', mec = 'black', label = 'y = 0')    # X[1] = X1
plt.plot(X[pos, 1], X[pos, 2], 's', color = 'yellow', mec = 'black', label = 'y = 1')  # X[2] = X2
plt.xlabel('Test 1 score')
plt.ylabel('Test 2 score')
plt.legend(loc = 'upper right')
plt.show(block = False)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_gradient_descent(X, Y, w_init, alpha, lamda, max_iter = 10000):
    w = w_init
    iter = 0;
    global m
    n = X.shape[1]
    while iter < max_iter:
        # Batch Gradient Descent
        error = sigmoid(X.dot(w)) - Y
        w = w - alpha * (1 / m) * X.T.dot(error) - lamda * (1 / (n - 1)) * w
        w[0] += lamda * (1 / (n - 1)) * w[0]  # No regularize bias term w[0]

        iter += 1
    return w

# Define parameters for Gradient Descent
alpha = 0.5  # Learning rate
lamda = 1    # Regularization term lambda
w_init = np.zeros((X.shape[1], 1))

# Training model
print('Training model using gradient descent...')
w = logistic_gradient_descent(X, Y, w_init, alpha, lamda)
print('Trained coefficients: \n', w[0::])

# Accuracy
Y_pred = np.floor(sigmoid(X.dot(w)) + 0.5)
print('Accuracy:', np.mean((Y_pred == Y).astype(float)) * 100)

def predict(X, w):
    X[:, 1::] = (X[:, 1::] - mu) / sd  # Feature Normalize
    return X.dot(w)

# Plot the decision boundary
# Take min, max of x1, x2 BEFORE normalizing
x1_min, x1_max = data[:, 0].min() - .5, data[:, 0].max() + .5
x2_min, x2_max = data[:, 1].min() - .5, data[:, 1].max() + .5
h = 0.05  # Step size in the mesh
xx1, xx2 = np.mgrid[x1_min:x1_max:h, x2_min:x2_max:h]
yy = predict(np.c_[map_feature(xx1.ravel(), xx2.ravel(), degree)], w)
yy = yy.reshape(xx1.shape)

xx1 = (xx1 - mu[0]) / sd[0]  # Normalize feature with corresponding mu, sd
xx2 = (xx2 - mu[1]) / sd[1]

plt.contour(xx1, xx2, yy, levels = [0], colors = 'green')
plt.draw()
