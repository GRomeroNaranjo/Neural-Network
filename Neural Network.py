import numpy as np

class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(input_size, 1)
    def forward(self, input):
        self.input = input
        output = np.dot(self.input, self.weights) + self.biases
        return output
    def backward(self, output_gradient, learning_rate):
       weights_gradient = np.dot(output_gradient, self.input.T)
       input_gradient = np.dot(output_gradient, self.weights.T)
       
       self.weights -= learning_rate * weights_gradient
       self.biases -= learning_rate * output_gradient
       
       return input_gradient
        