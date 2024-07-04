import numpy as np

class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(input_size, 1)
    def forward(self, input):
        output = np.dot(input, self.weights) + self.biases
        return output
    def backward(self, input):
        pass