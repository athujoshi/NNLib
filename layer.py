import numpy as np
import neuron

class denselayer():
    
    def __init__(self,no_of_neurons,input_dim,activation="None"):

        self.input_dim=input_dim
        self.output_dim=no_of_neurons
        self.neurons=neuron.neuron_stack(self.input_dim,no_of_neurons,activation=activation)

    def forward(self,input_data):

        return self.neurons.forward(input_data)

    def backprop(self,der,learning_rate):
        return self.neurons.backprop(der,learning_rate)





"""
class denselayer():
    
    def __init__(self,no_of_neurons,input_dim):

        self.neurons=[]
        self.input_dim=input_dim
        self.output_dim=no_of_neurons

        for _ in range(no_of_neurons):
            self.neurons.append(neuron.neuron(self.input_dim))

    def forward(self,input_data):

        output_data=[]

        for i in range(len(self.neurons)):

            output_data.append(self.neurons[i].forward(input_data))

        return np.concatenate(output_data,axis=1)

    #def backprop(self)
"""

