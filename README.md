

## **Project Description**

This project aims to develop a simple streamlined neural network with very few lines of code. It portrays how AI research and math comes together into developing an ANN. This neural network implements many mathematical conecpts such as lorgarithms, chain rule... In order to correctly implement a functional neural network.

## **Testing NN**

This neural network will be used to tackle the mnist digit classification. Mnist is a 70,000 image dataset of handwritten images each 28 by 28 pixels. Mnist stands for the Modified National Institute of Standards and Technology database, and it is mainly popularised for training deep learning models.

## **Code Implementation**

Below you will find how you can implement this neural network based on our OOP programming:



```
dense1 = Layers_Dense(784, 128)
activation1 = Relu()

dense2 = Layers_Dense(128, 10)
activation2 = Softmax()

learning_rate = 0.1
epochs = 250

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
```



