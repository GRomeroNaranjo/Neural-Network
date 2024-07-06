from tensorflow.keras import datasets
from Neural_Network import Layers_Dense, Relu, Softmax, sparse_categorical_crossentropy, predict
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

dense1 = Layers_Dense(784, 128)
activation1 = Relu()

dense2 = Layers_Dense(128, 10)
activation2 = Softmax()

learning_rate = 0.1
epochs = 1250

for epoch in range(epochs):
    dense1.forward(X_train)
    activation1_output = activation1.forward(dense1.output)

    dense2.forward(activation1_output)
    activation2_output = activation2.forward(dense2.output)

    error = sparse_categorical_crossentropy(activation2_output, y_train)
    
    predictions = np.argmax(activation2_output, axis=1)
    accuracy = np.mean(predictions == y_train)

    print(f'Epoch {epoch+1}, Error: {error}, Accuracy: {accuracy}')

    dvalues = activation2_output
    dvalues[range(len(dvalues)), y_train] -= 1
    dvalues = dvalues / len(dvalues)

    dinputs = dense2.backward(dvalues, learning_rate)
    dinputs = activation1.backward(dinputs)
    dense1.backward(dinputs, learning_rate)
    
    


