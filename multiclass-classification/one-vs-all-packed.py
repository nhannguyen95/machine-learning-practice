import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio
import random
from sklearn import linear_model

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
    # Specify figsize to avoid unexpected gap between cells
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

# C is the inverse of lambda: the higher C is, the more
# the algorithm will focus on the fitting data term
logreg = linear_model.LogisticRegression(C = 1)

print('Training model (this may take a while)...')
logreg.fit(X, y.ravel())

# Predict and compute accuracy
y_pred = logreg.predict(X)
y_pred = y_pred.reshape((m, 1))
print('Accuracy: ', np.mean((y_pred == y).astype(float)) * 100)
