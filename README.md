# NNLib

ANN Library for Regression and Classification created from scratch using only NumPy.

Project created to understand modular implementation of BackPropogation in a library.

Implemented Activation Functions : Sigmoid, ReLu and Softmax.

Implemented Loss Functions : Mean Squared Error, Multi Class Cross Entropy.

#Sample Implementation

import nn

k=nn.nn()

k.add_layer(no_of_neurons=4,dim=2,activation="sigmoid")

k.add_layer(no_of_neurons=4,activation="sigmoid")

k.add_layer(no_of_neurons=2,activation="softmax")

k.define_loss_function("cross_entropy")

k.backprop(x,y,epochs=100000,every_n_epoch=10000,learning_rate=0.01)

predicted=k.forward(x)

