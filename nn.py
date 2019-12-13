import layer
import numpy as np
import loss_functions

class nn():

    def __init__(self,model="Sequential"):
        """initialise a neural network with given model"""
        
        self.model=model
        self.layers=[]

    def add_layer(self,no_of_neurons=None,input_dim=None,activation="None"):
        """add a layer of neurons"""

        if no_of_neurons==None:
            raise ValueError("Please provide number of neurons")

        if len(self.layers)==0:
            if input_dim==None:
                raise ValueError("Please provide input dim!")
            
            else:
                self.layers.append(layer.denselayer(no_of_neurons,input_dim,activation=activation))

        else:
            input_dim=self.layers[len(self.layers)-1].output_dim
            self.layers.append(layer.denselayer(no_of_neurons,input_dim,activation=activation))

    def define_loss_function(self,loss_function="mean_squared_error"):
        """iniates a loss_function object corresponding to mentioned loss_function"""
        
        self.loss_functions=loss_functions.loss_functions(loss_function=loss_function)

    def forward(self,input_data):
        """forward calculates for x(input)"""

        output=input_data
        
        for k in range(len(self.layers)):
            
            output=self.layers[k].forward(output)

        return output

    def backprop(self,input_data,to_predict,epochs,every_n_epoch,learning_rate):
        """Backpropagation to update weights and biases of network"""

        for epoch in range(epochs):
            
            predicted = self.forward(input_data)

            der = self.loss_functions.calculate_loss(predicted=predicted,to_predict=to_predict)
            """
            if np.isnan(der).any():
                continue
            """
            #print loss
            self.loss_functions.print_loss(epoch=epoch,every_n_epoch=every_n_epoch)
            
            """
            # relative error module 
            
            if len(self.mse_loss_list)>0:
                last_mse=self.mse_loss_list[len(self.mse_loss_list)-1]
                print(last_mse,mse_loss)
                perc_relative_error=np.absolute((last_mse-mse_loss)*100/last_mse)
                if perc_relative_error < 0.0001 :
                    break
            """
            
            for layer in range(len(self.layers)):
                der=self.layers[len(self.layers)-layer-1].backprop(der,learning_rate)
        



