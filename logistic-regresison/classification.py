import numpy as np
import matplotlib.pyplot as plt

# The data is from Coursera Machine Learning course/week3/ex2data1.txt
data = np.loadtxt('../data/data3.txt', delimiter = ',')

m = data.shape[0]  # Number of examples
n = data.shape[1]  # Number of columns

X = data[:, 0:(n - 1)]          # Score of first and second exam
Y = data[:, np.newaxis, n - 1]  # Pass or fail

# Feature Normalization using Standardization
# x = (x - μ) / σ
mu = np.mean(X, axis = 0)
sd = np.std(X, axis = 0)
X = (X - mu) / sd

X = np.column_stack((np.ones((X.shape[0], 1)), X))  # Add bias term (e.g. X[0] = 1)

# Visualize data
print('Plotting the data...')
plt.figure(1)
plt.grid()
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
neg = np.where(Y == 0)[0]  # index of fail examples
pos = np.where(Y == 1)[0]  # index of pass examples
plt.plot(X[neg, 1], X[neg, 2], 'o', color = 'blue', mec = 'black', label = 'Fail')
plt.plot(X[pos, 1], X[pos, 2], 's', color = 'yellow', mec = 'black', label = 'Pass')
plt.legend(loc = 'upper right')
plt.show(block = False)

input('Press enter to continue...')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_gradient_descent(X, Y, w_init, alpha, max_iter = 200):
    w = w_init
    iter = 0
    global m
    while iter < max_iter:
        # Batch gradient descent
        error = sigmoid(X.dot(w)) - Y
        w = w - alpha * (1 / m) * X.T.dot(error)

        iter += 1
    return w

# Define parameter for Gradient Descent
alpha = 0.0025  # Learning rate
w_init = np.zeros((X.shape[1], 1))

# Training model
print('Training model using gradient descent...')
w = logistic_gradient_descent(X, Y, w_init, alpha)
print('Trained coefficients: \n', w[0::])

# Plot the decision boundary
print('Plotting the decision boundary...')
plt.plot(X[:, 1], (-w[0] - w[1] * X[:, 1]) / w[2],
                    color = 'green',
                    linewidth = '0.5')
plt.draw()

# Make prediction on new data
print('Now we will decide if a student passes or fails \n \
       given the score of his two exams')
score1 = int(input('Enter exam 1 score: '))
score2 = int(input('Enter exam 2 score: '))
score1 = (score1 - mu[0]) / sd[0]  # Feature Normalization
score2 = (score2 - mu[1]) / sd[1]
if sigmoid((w[0] + w[1] * score1 + w[2] * score2)) >= 0.5:
    print('Pass')
else:
    print('Fail')
