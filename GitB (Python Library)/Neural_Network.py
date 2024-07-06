import numpy as np
from tensorflow.keras import datasets
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

class Layers_Dense():
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        return self.dinputs

class Relu():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Softmax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs

def sparse_categorical_crossentropy(y_pred, y_true):
    samples = len(y_true)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_confidences = y_pred_clipped[range(samples), y_true]
    negative_log_likelihoods = -np.log(correct_confidences)
    return np.mean(negative_log_likelihoods)

def accuracy(y_pred, y_true):
  predictions = np.argmax(y_pred, axis=1)
  accuracy = np.mean(predictions == y_true)
  return accuracy

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output