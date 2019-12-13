import numpy as np

class loss_functions():

    def __init__(self,loss_function="mean_squared_error"):
        self.loss_function=loss_function
        self.loss_list=[]

    def calculate_loss(self,predicted,to_predict):

        if self.loss_function=="mean_squared_error":
            return self.mean_squared_error(predicted,to_predict)

        elif self.loss_function=="cross_entropy":
            return self.cross_entropy(predicted,to_predict)

        else:
            raise ValueError("Invalid Loss Function/ Loss Function not defined")

    def print_loss(self,epoch,every_n_epoch):
        if epoch%every_n_epoch==0:
            print(self.loss_function+" at epoch "+str(epoch+1)+" is "+str(self.loss_list[len(self.loss_list)-1]))

    def mean_squared_error(self,predicted,to_predict):
        se_loss=((to_predict-predicted)**2)*0.5
        loss=np.average(se_loss)
        self.loss_list.append(loss)
        der=-1*(to_predict-predicted)
        return der

    def cross_entropy(self,predicted,to_predict):
        ce_loss=-1*(np.sum(np.multiply(np.log(predicted),to_predict),axis=1))
        loss=np.average(ce_loss)
        self.loss_list.append(loss)
        der=-1*np.divide(to_predict,predicted)
        return der

    








