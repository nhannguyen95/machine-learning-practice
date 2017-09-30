import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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

# C is the inverse of lambda: the higher C is, the more
# the algorithm will focus on the fitting data term
logreg = linear_model.LogisticRegression(C = 1)

# We create an instance of Neighbours Classifier and fit the data
print('Training model...')
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to
# each point in the mesh [x_min, x_max]x[y_min, y_max]
h = .02  # Step size in the mesh
x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[map_feature(xx.ravel(), yy.ravel(), degree)])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Paired)

# Plot also the training points.
# cmap is a colormap, Paired means it only has 2 colors (0, 1).
# Each point is assigned a value in c array, this value is used
# to color the point by mapping to cmap.
plt.scatter(X[:, 1], X[:, 2], c = Y, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('Test 1 score')
plt.ylabel('Test 2 score')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show(block = False)
