import numpy as np

class neuron_stack():

    def __init__(self,dim,no_of_neurons,activation="None"):
        
        self.activation=activation
        self.no_of_neurons=no_of_neurons
        self.weights = np.random.uniform(low=-1., high=1., size=(dim,int(self.no_of_neurons)))
        self.bias = np.random.uniform(low=-1., high=1., size=(self.no_of_neurons,1)).T
    
    def sigmoid(self,input_data,forward=True):
        
        if forward:
            return np.exp(input_data)/((np.exp(input_data))+1)

        else:
            return (input_data) * (1-input_data)

    def relu(self,input_data,forward=True):

        if forward:
            return np.maximum(input_data,0)
        
        else:
            input_data[input_data<=0] = 0
            input_data[input_data>0] = 1

    def softmax(self,input_data,forward=True):

        if forward:
            x_exp=np.exp(input_data)
            x_exp_sum=np.repeat(np.sum(x_exp,axis=1)[:, np.newaxis],len(x_exp.T),axis=1)
            return x_exp/x_exp_sum

        else:
            x_exp=np.exp(self.x_star)
            x_exp_sum=np.repeat(np.sum(x_exp,axis=1)[:, np.newaxis],len(x_exp.T),axis=1)
            #return (x_exp*(1-(x_exp/x_exp_sum)))/x_exp_sum
            x_sm=x_exp/x_exp_sum
            #print(input_data)
            #print(x_sm)
            der_sum_xs=np.repeat(np.sum(input_data*x_sm,axis=1)[:, np.newaxis],len(x_sm.T),axis=1)
            #print(der_sum_xs)
            #print(np.multiply(x_sm,(input_data-der_sum_xs)))
            return np.multiply(x_sm,(input_data-der_sum_xs))
    
    def forward(self, input_data):
        
        self.x=input_data
        biases = np.repeat(self.bias,len(self.x),axis=0)
        self.x_star=(self.x.dot(self.weights))+biases
        
        if self.activation=="sigmoid":
            self.op = self.sigmoid(self.x_star,forward=True)
        elif self.activation=="relu":
            self.op = self.relu(self.x_star,forward=True)
        elif self.activation=="softmax":
            self.op = self.softmax(self.x_star,forward=True)
        else:
            self.op=self.x_star

        return self.op

    def backprop(self,der,learning_rate=0.01):
        
        if self.activation=="sigmoid":
            #der= self.op*(1-self.op)*der
            der= self.sigmoid(self.op,forward=False)*der
        elif self.activation=="relu":
            try:
                der= self.relu(self.op,forward=False)*der
            except:
                pass
        elif self.activation=="softmax":
            try:
                der= self.softmax(der,forward=False)
            except:
                pass
        else:
            pass
        
        d_down=der.dot(self.weights.T)
        dw=(self.x.T).dot(der)
        self.weights=self.weights-(learning_rate*dw)
        db=np.sum(der,axis=0).T
        self.bias=self.bias-(learning_rate*db)
        
        return d_down




"""
class neuron():
    
    def __init__(self,dim):

        self.weights=np.random.uniform(low=-1., high=1., size=(dim,1))
        self.bias=np.random.uniform(-1,1)

    def forward(self,input_data):
        self.x=input_data
        return self.sigmoid(np.dot(input_data,self.weights)+self.bias,forward=True)

    def sigmoid(self,input_data,forward=True):

        if forward:
            return np.exp(input_data)/((np.exp(input_data))+1)

        else:
            return (input_data) * (1-input_data)
    
"""