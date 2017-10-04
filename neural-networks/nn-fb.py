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

# Hard code the number of layers' units of neural network
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidGrad(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Implement Backpropagation to compute gradient of coefficient
def computeGrad(nn_theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    # Reshape nn_coefs back into the parameters theta1 and theta2,
    # the weight matrices for our 2 layer neural network.
    theta1 = nn_theta[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    theta2 = nn_theta[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # Useful variables
    m = X.shape[0]

    # Add bias term
    X = np.column_stack((np.ones((m, 1)), X))

    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((num_labels, hidden_layer_size + 1))

    for t in range(m):
        curX = X[t, :, np.newaxis]
        curY = (np.arange(1, num_labels + 1) == y[t]).astype(int)
        curY = np.array([curY]).T  # row vector to column vector, 10x1

        # Feedforward propagation
        a1 = curX  # 401 x 1
        z2 = theta1.dot(a1)  # 25 x 1
        a2 = sigmoid(z2)
        a2 = np.row_stack((np.ones((1, a2.shape[1])), a2))  # 26 x 1
        z3 = theta2.dot(a2)  # 10 x 1
        a3 = sigmoid(z3)

        # Backpropagation
        delta3 = a3 - curY  # 10 x 1
        delta2 = theta2.T.dot(delta3)[1:] * sigmoidGrad(z2)  # 25 x 1

        Delta1 += delta2.dot(a1.T)  # 25 x 401
        Delta2 += delta3.dot(a2.T)  # 10 x 26

    theta1_grad = Delta1 / m;
    theta2_grad = Delta2 / m;

    # Regularization
    theta1_grad += lamda / m * np.column_stack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    theta2_grad += lamda / m * np.column_stack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

    grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()), axis = 0)  # row vector
    grad = np.array([grad]).T
    return grad

def gradient_descent(X, y, nn_theta_init, alpha, lamda,
                     input_layer_size, hidden_layer_size, num_labels, max_iter = 200):
    nn_theta = nn_theta_init

    iter = 0
    while iter < max_iter:
        nn_theta = nn_theta - alpha * computeGrad(nn_theta,
                                                  input_layer_size,
                                                  hidden_layer_size,
                                                  num_labels,
                                                  X, y, lamda)
        print(max_iter - iter - 1)
        iter += 1

    return nn_theta

def randInitTheta(L_in, L_out):
    eps_init = np.sqrt(6 / (L_in + L_out))  # Compute the recommend value for eps
    return np.random.rand(L_out, L_in + 1) * 2 * eps_init - eps_init

# Some parameters for Gradient Descent
theta1_init = randInitTheta(input_layer_size, hidden_layer_size)
theta2_init = randInitTheta(hidden_layer_size, num_labels)
nn_theta_init = np.concatenate((theta1_init.ravel(), theta2_init.ravel()), axis = 0)  # row vector
nn_theta_init = np.array([nn_theta_init]).T  # row vector to column vector
alpha = 0.05  # Learning rate
lamda = 0     # Regularization term

# Training out neural network using Gradient Descent
print('Traning neural network (this could take a while)...')
nn_theta = gradient_descent(X, y, nn_theta_init, alpha, lamda,
                            input_layer_size, hidden_layer_size, num_labels)

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

# Predict
print('Predicting...')
theta1 = nn_theta[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
theta2 = nn_theta[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))
y_pred = predict(theta1, theta2, X)

# Accuracy
print('Accuracy: ', np.mean((y_pred == y).astype(float)) * 100)
