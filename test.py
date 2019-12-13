
import matplotlib.pyplot as plt

import numpy as np

import neuron
import layer
import nn


jk=[1,2,3,4,34,54]
jk.pop(4)

k=nn.nn()


k.add(1,1)
k.add(5)
k.add(3)
k.add(1)

x=np.array([[.1,.2,.3]]).T
y=np.array([[.2,.4,.6]]).T



input_data=np.random.rand(500,3)
k.forward(x)

k.backprop(x,y)

for _ in range(1000):
    k.backprop(x,y)


f=np.array([[0.3,0.2,0.5,0.1,0.1],[0.2,0.4,0.6,0.1,0.1],[0.5,0.2,0.3,0.1,0.1],[0.6,0.4,0.2,0.1,0.2]]).T
g=np.array([[0.8,0.6,0.8,0.2,0.25],[0.4,0.3,0.4,0.1,0.125]]).T


np.isnan(f).any()


test=np.array([[0.2,0.4]])

#even odd dataset
x=np.random.randint(low=0, high=101, size=(100000,2))
y=np.sum(x,axis=1)[:, np.newaxis]
y[y%2!=0]=0
y[y!=0]=1
y_star=1-y
y_final=np.concatenate((y,y_star),axis=1)
y=y_final
x=x/100.

import nn

k=nn.nn()
k.add_layer(4,2,activation="sigmoid")
k.add_layer(4,activation="sigmoid")
k.add_layer(2,activation="softmax")

k.define_loss_function("cross_entropy")

losss=k.loss_functions.loss_list
plt.plot(losss)

k.layers[0].neurons.weights
k.layers[1].neurons.weights

test=np.array([[80,49]])

k.forward(test)

k.forward(f)
k.backprop(f,g,1000000,100000,0.01)
k.backprop(f,g,2,1,0.01)

k.backprop(x,y,100000,10000,0.01)
y_hat=k.forward(x)

np.sum(y,axis=0)



x_soft=k.forward(f)
x_soft=np.exp(x_soft)

x_back=(1-(x_soft/x_star))/x_star

x_star=np.repeat(np.sum(x_soft,axis=1)[:, np.newaxis],len(x_soft.T),axis=1)

x_soft/x_star

mlist=k.mse_loss_list
plt.plot(mlist)








predicted=k.forward(f)
to_predict=g

se_loss=((to_predict-predicted)**2)*0.5
mse_loss=np.average(se_loss)


der=-1*(to_predict-predicted)

fx=k.layers[2].neurons.op
fxx=1-fx

x=k.layers[2].neurons.x

yups=der*fx*fxx

tyu=np.multiply(x,yups)


dw=(x.T).dot(yups)

weights=k.layers[2].neurons.weights-0.01*dw
k.layers[1].neurons.bias

np.sum(yups,axis=0).T

yups.dot(k.layers[2].neurons.weights.T)




import nn
import quandl#FOR GETTING DATA
quandl.ApiConfig.api_key = "6nvvFYs2phJ88CagMKHd"


edelweiss_data=quandl.get("NSE/NIFTYEES")
edelweiss_data["to_predict"]=edelweiss_data["Close"].shift(-1)



x=np.array((edelweiss_data.iloc[0:472,0:4]))
y=np.array(([edelweiss_data.iloc[0:472,7]])).T


k=nn.nn()
k.add(15,4,activation='relu')
k.add(15,activation='relu')
k.add(5,activation='relu')
k.add(1)


x.fillna(0)

predicted=x.dot(k.layers[0].neurons.weights)+np.repeat(k.layers[0].neurons.bias,len(x),axis=0)

se_loss=((y-predicted)**2)*0.5
der=-1*(y-predicted)


dw=(x.T).dot(der)


predicted=k.forward(x)

plt.plot(predicted,'r')
plt.plot(y,'b')
plt.show()

test=x[0,:]

k.forward(np.array(x[0,:]))
k.backprop(x,y,3000000,500000,0.0000000000001)



































test=np.repeat(np.random.uniform(low=-1., high=1., size=(2,1)).T,3,axis=0)

ret=h+test

np.exp(ret)/((np.exp(ret))+1)

ret*(1-ret)

ret=np.repeat(test, 3, axis=0)



layers=k.layers

op1=k.layers[0].forward(input_data)
op2=k.layers[1].forward(op1)
op3=k.layers[2].forward(op2)

sampl = np.random.uniform(low=-1., high=1., size=(5,1))
weights=np.random.rand(5,1)

s=np.random.rand(3,1)
w=np.random.rand(3,1)

float(np.shape(s)[0])

float(0.5*sum((s-w)**2))







k=layer.denselayer(4,3)

input_data=np.random.rand(5,3)

ops=[]
for i in range(len(k.neurons)):
    ops.append(k.neurons[i].forward(input_data))

s=np.concatenate(ops,axis=1)

h=np.array(ops)

s=np.empty((5,1))


net=nn.nn()
net.add(4,3)

layers=net.layers


neurons=k.neurons


f=np.array([[3,2,5],[2,4,6],[5,2,3],[6,4,2]]).T
g=np.array([[7,4,6,9]]).T


net=nn.nn()

net.add(3,4)
net.add(1)

net.forward(f)



