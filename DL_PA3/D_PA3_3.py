

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def softmax(x):
    return np.exp(x) / (np.sum(np.exp(x)))

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

###########################################################################
class hiddenlayer2 : ##hidden layer 개수 2
    def init_network(): ## 첫번째 hidden layer의 node 수 100, 두번째 hidden layer의 node 수 30
        network = {}
        network['w1'] = np.random.normal(0,np.sqrt(2/(784+100)),[784,100]) ##xavier initialization
        ##network['w1'] = np.ones((784,100))*0.05
        network['b1'] = np.ones((1,100))*0.2
        network['w2'] = np.random.normal(0,np.sqrt(2/(100+30)),[100,30]) ##xavier initialization
        ##network['w2'] = np.ones((100,30))*0.1
        network['b2'] = np.ones((1,30))*0.2
        network['w3'] = np.random.normal(0,np.sqrt(2/(30+10)),[30,10]) ##xavier initialization
        ##network['w3'] = np.ones((30,10))*0.1
        network['b3'] = np.ones((1,10))*0.2
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
        y = softmax(a3) ## output layer의 활성홤수는 softmax 
        
        ##backward
        delta3 = y - t ## activation function이 softmax, error functino이 CECF일 경우, delta값은 y-t
        delta2 = np.dot(delta3,w3.T)*z2*(1-z2)
        delta1 = np.dot(delta2,w2.T)*z1*(1-z1)
        ##update
        lr = 0.2 ##learning rate
        b3 -= lr*delta3
        w3 -= lr*np.dot(z2.T,delta3)
        b2 -= lr*delta2
        w2 -= lr*np.dot(z1.T,delta2)
        b1 -= lr*delta1
        w1 -= lr*np.dot(x.T,delta1)
        temp={}
        temp['w1'], temp['w2'], temp['w3'] = w1, w2, w3
        temp['b1'],temp['b2'],temp['b3'] = b1, b2, b3
        return temp

    def test(network, x, t):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = softmax(a3)
        predict =  np.argmax(y[0])
      
        if predict == t : ## predict와 target이 같으면 예측 성공, 다르면 예측 실패로 간주.
            error = 0
        else:
            error = 1
        return error
#############################################################################

#input
training_data = 200 ### 2 : 1 : 1 
val_data = 100
test_data = 100
epoch =500

network = hiddenlayer2.init_network()
df = pd.read_csv('mnist_400.csv', encoding='utf-8', header = None)
######################### training #######################################
for k in range(epoch):
    for i in range(training_data+val_data):
        x = np.array(df.loc[[i],1:])/255
        target = int(df.loc[[i],0])
        t_vector = np.zeros((1,10))
        t_vector[0][target] = 1
        network = hiddenlayer2.training(network,x,t_vector)  

####################### test ###############################################
total_error = 0    
for i in range(training_data+val_data,training_data+val_data+test_data): ##test
    x = np.array(df.loc[[i],1:])/255
    target = int(df.loc[[i],0])
    error = hiddenlayer2.test(network,x,target) 
    total_error += error

error_rate = float(total_error / test_data*100)
print('error = ',error_rate,'%')

