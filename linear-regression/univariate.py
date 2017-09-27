import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# The data is from Coursera Machine Learning course/week2/ex2data.txt
data = np.loadtxt('../data/data1.txt', delimiter=',')

X = data[:, np.newaxis, 0]  # Population
Y = data[:, np.newaxis, 1]  # Profit

# Visualize data
plt.figure()
plt.grid()
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X, Y, 'r+')

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model
regr.fit(X, Y)

# Make predictions
Y_pred = regr.predict(X)

# The coefficients
print('Coefficients: %f %f\n' % (regr.intercept_, regr.coef_))

# Plot output
plt.plot(X, Y_pred, color='blue')
plt.show()
