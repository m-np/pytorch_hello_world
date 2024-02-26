"""This code looks at different activation functions"""
import numpy as np
import matplotlib.pyplot as plt

def plot_func(x,y, title):
    """Plots activation function"""
    # helper function to plot activation functions
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('activation(x)')
    plt.grid(True)
    plt.show()


def sigmoid(x):
    """sigmoid activation"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Relu activation function"""
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    """Leaky Relu activation function"""
    return np.maximum(alpha*x, x)


def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_prime(x):
    """Derivative function for tanh"""
    return 1-np.tanh(x)**2


def softmax(x):
    """Softmax activation function"""
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)


def mse(y_true, y_pred):
    """MSE Loss"""
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    """Derivative of loss function"""
    return 2*(y_pred-y_true)/y_true.size


if __name__ == "__main__":

    x = np.linspace(-10, 10, 100)

    plot_func(x, sigmoid(x), "sigmoid")

    plot_func(x, relu(x), "relu")

    plot_func(x, leaky_relu(x), "leaky_relu")

    plot_func(x, tanh(x), "tanh")

    plot_func(x, softmax(x), "softmax")