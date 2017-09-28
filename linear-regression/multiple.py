import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

# The data is from Coursera Machine Learning course/week2/ex2data2.txt
data = np.loadtxt('../data/data2.txt', delimiter=',')

n = data.shape[1]  # Number of features

X = data[:, 0:(n - 1)]          # Size of house and number of bedrooms
Y = data[:, np.newaxis, n - 1]  # Price of the house

# Feature Normalization using Standardization
# x = (x - μ) / σ
mu = np.mean(X, axis = 0)
sd = np.std(X, axis = 0)
X = (X - mu) / sd

# Visualize the data (since we know the data2.txt is in 3D)
print('Plotting the data...')
fig = plt.figure(1)
fig.canvas.set_window_title('Data visualization before Feature Normalization')
ax = fig.gca(projection = '3d')
ax.scatter(X[:, np.newaxis, 0], X[:, np.newaxis, 1], Y, color = 'red', marker = '+')
ax.set_xlabel('Size of house')
ax.set_ylabel('Number of bedrooms')
ax.set_zlabel('Price of house')
plt.show(block = False)

input("Press enter to continue...")

# Create linear regression object
regr = linear_model.LinearRegression()

# Training
print('Training the linear model...')
regr.fit(X, Y)

# The coefficients
print('Intercept: ', regr.intercept_)
print('Coefficients: ', regr.coef_)

# Plot the separating hyperlane
print('Plotting the separating hyperlane...')
X1_grid = np.array([np.min(X[:, np.newaxis, 0]), np.max(X[:, np.newaxis, 0])])
X2_grid = np.array([np.min(X[:, np.newaxis, 1]), np.max(X[:, np.newaxis, 1])])
X1_grid, X2_grid = np.meshgrid(X1_grid, X2_grid)
Y_grid = regr.intercept_ + regr.coef_[0][0] * X1_grid + regr.coef_[0][1] * X2_grid
ax.plot_surface(X1_grid, X2_grid, Y_grid,
                    linewidth = 0,
                    antialiased = False,
                    color = (0.118, 0.565, 1.000, 0.5))
plt.draw()

# Make predictions on new data
print('Now we will predict the price of a new house')
X1 = float(input('Enter the size of house: '))
X2 = float(input('Enter the number of bedrooms: '))
X1 = (X1 - mu[0]) / sd[0]  # Feature Normalization
X2 = (X2 - mu[1]) / sd[1]
Y_pred = regr.intercept_ + regr.coef_[0][0] * X1 + regr.coef_[0][1] * X2
print('The predicted price is: ', Y_pred)
