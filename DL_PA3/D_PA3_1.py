
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

def setting_data(data):
    
    ##case of normalized (input data, output data)
    crim = np.array(data.loc[:,['CRIM']])
    data['CRIM'] = normalized(crim)
    zn = np.array(data.loc[:,['ZN']])
    data['ZN'] = normalized(zn)
    indus = np.array(data.loc[:,['INDUS']])
    data['INDUS'] = normalized(indus)
    nox = np.array(data.loc[:,['NOX']])
    data['NOX'] = normalized(nox)
    rm = np.array(data.loc[:,['RM']])
    data['RM'] = normalized(rm)
    age = np.array(data.loc[:,['AGE']])
    data['AGE'] = normalized(age)
    dis = np.array(data.loc[:,['DIS']])
    data['DIS'] = normalized(dis)
    rad = np.array(data.loc[:,['RAD']])
    data['RAD'] = normalized(rad)
    tax = np.array(data.loc[:,['TAX']])
    data['TAX'] = normalized(tax)
    p = np.array(data.loc[:,['PTRATIO']])
    data['PTRATIO'] = normalized(p)
    b = np.array(data.loc[:,['B-1000']])
    data['B-1000'] = normalized(b)
    lstat = np.array(data.loc[:,['LSTAT']])
    data['LSTAT'] = normalized(lstat)
    med = np.array(data.loc[:,['MEDV']])
    data['MEDV'] = normalized(med)
    
    return data
#########################################################################
class hiddenlayer1 : ##hidden layer 개수 1
    def init_network(): ##hidden layer의 node 수 20
        i = 13
        h = 20
        o = 1
        network = {}
        network['w1'] = np.ones((i,h))*0.1
        network['b1'] = np.ones((1,h))*0.01
        network['w2'] = np.ones((h,o))*0.1
        network['b2'] = np.ones((1,o))*0.01
        return network

    def training(network, x, t):
        ##forward
        w1, w2 = network['w1'], network['w2']
        b1, b2 = network['b1'], network['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = identity_function(a2)
       
        ##backward
        delta2 = y - t
        delta1 = np.dot(delta2,w2.T)*z1*(1-z1)
        ##update
        lr = 0.1 ##learning rate
        b2 -= lr*delta2
        w2 -= lr*np.dot(z1.T,delta2)
        b1 -= lr*delta1
        w1 -= lr*np.dot(x.T,delta1)
        temp={}
        temp['w1'], temp['w2'] = w1, w2
        temp['b1'],temp['b2'] = b1, b2
        return temp

    def test(network, x, t):
        w1, w2 = network['w1'], network['w2']
        b1, b2 = network['b1'], network['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = identity_function(a2)
        error = 0.5*( y - t)* (y - t)
        return error
###########################################################################
class hiddenlayer2 : ##hidden layer 개수 2
    def init_network(): ## 첫번째 hidden layer의 node 수 10, 두번째 hidden layer의 node 수 5
        network = {}
        i=13
        h1 = 10
        h2 = 5
        o = 1
        network['w1'] = np.ones((i,h1))*0.1
        network['b1'] = np.ones((1,h1))*0.01
        network['w2'] = np.ones((h1,h2))*0.1
        network['b2'] = np.ones((1,h2))*0.01
        network['w3'] = np.ones((h2,o))*0.1
        network['b3'] = np.ones((1,o))*0.01
        return network

    def training(network, x, t):
        ##forward
        w1, w2, w3 =network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = identity_function(a3)
        
        error = 0.5*( y - t)* (y - t)
        ##backward
        delta3 = y - t
        delta2 = np.dot(delta3,w3.T)*z2*(1-z2)
        delta1 = np.dot(delta2,w2.T)*z1*(1-z1)
        ##update
        lr = 0.1 ##learning rate
        b3 -= lr*delta3
        w3 -= lr*np.dot(z2.T,delta3)
        b2 -= lr*delta2
        w2 -= lr*np.dot(z1.T,delta2)
        b1 -= lr*delta1
        w1 -= lr*np.dot(x.T,delta1)
        temp={}
        temp['w1'], temp['w2'], temp['w3'] = w1, w2, w3
        temp['b1'],temp['b2'],temp['b3'] = b1, b2, b3
        return temp, error

    def test(network, x, t):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = identity_function(a3)
        error = 0.5*( y - t)* (y - t)
        return error
#############################################################################

#input
training_data = 404### 8 : 1 : 1
val_data = 50
test_data = 50
epoch = 200

##network = hiddenlayer1.init_network() ##hidden layer 1개인 경우
network = hiddenlayer2.init_network() ##hidden layer 2개인 경우

df = pd.read_csv('boston_housing.csv', encoding='utf-8')
sdf = setting_data(df) ##normalized data

######################### training #######################################
num_train = np.arange(0,(training_data+val_data)*epoch,1)
train_error = np.array([])
t=0
for k in range(epoch):
    for i in range(training_data+val_data):
        x = np.array(sdf.loc[[i],'CRIM':'LSTAT'])
        target = np.array(sdf.loc[[i],'MEDV'])
        ##network = hiddenlayer1.training(network,x,target) ##hidden layer 1개인 경우
        network,error = hiddenlayer2.training(network,x,target)  ##hidden layer 2개인 경우
        train_error = np.append(train_error, error)
        t +=1

plt.title('error function(train)')
plt.plot(num_train,train_error)
plt.show()

####################### test ###############################################
num_test = np.arange(0,test_data,1)
test_error = np.array([])
t=0

for i in range(training_data+val_data,training_data+val_data+test_data): ##test
    x = np.array(sdf.loc[[i],'CRIM':'LSTAT'])
    target = np.array(sdf.loc[[i],'MEDV'])
    error = hiddenlayer2.test(network,x,target)
    test_error = np.append(test_error,error)
    t+1
    
plt.title('error function(test)')
plt.plot(num_test,test_error)
plt.show()


