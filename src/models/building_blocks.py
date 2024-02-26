"""This code looks at different activation functions"""
import numpy as np
import matplotlib.pyplot as plt
from layers import Layer

class Linear(Layer):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            seed = 7):
        
        super().__init__()
        np.random.seed(seed)
        self.w = np.random.rand(in_dim, out_dim) - 0.5
        self.b = np.random.rand(1, out_dim) - 0.5
        # print(self.w)
        # print(self.b)

    def forward(self, input_batch):
        self.input = input_batch
        self.out = np.dot(input_batch, self.w) + self.b
        return self.out
    
    def backward(self, output_error, lr):
        # print(self.input, self.input.T)
        # print("oe", output_error)
        # if self.input
        weights_error = self.input.T*output_error
        # input_error = np.dot(output_error, self.w.T)
        bias_error = output_error
        # print(weights_error)

        self.w -= lr*weights_error.T
        self.b -= lr*bias_error
        # print(1, self.w)
        # print(1, self.b)


def Activation(Layer):
    def __init__(self, activation_fn, activation_prime):
        self.activation_fn = activation_fn
        self.activation_prime = activation_prime

    def forward(self, input_batch):
        self.input = input_batch
        self.out = self.activation_fn(input_batch)
        return self.out
    
    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, lr):
        return self.activation_prime(self.input) * output_error



if __name__ == "__main__":

    in_dim = 4
    out_dim = 1
    input_batch = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([[1]])
    lr = 0.0005

    l = Linear(in_dim, out_dim)
    out = l.forward(input_batch)
    output_error = out - y
    l.backward(output_error, lr)
