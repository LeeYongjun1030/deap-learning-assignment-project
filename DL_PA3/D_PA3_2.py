
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

def setting_data(data): ## for normalizing data
    age = np.array(data.loc[:,['age']])
    data['age'] = normalized(age)
    cp = np.array(data.loc[:,['cp']])
    data['cp'] = normalized(cp)
    trestbps = np.array(data.loc[:,['trestbps']])
    data['trestbps'] = normalized(trestbps)
    chol = np.array(data.loc[:,['chol']])
    data['chol'] = normalized(chol)
    fbs = np.array(data.loc[:,['fbs']])
    data['fbs'] = normalized(fbs)
    restecg = np.array(data.loc[:,['restecg']])
    data['restecg'] = normalized(restecg)
    thalach = np.array(data.loc[:,['thalach']])
    data['thalach'] = normalized(thalach)
    ca = np.array(data.loc[:,['ca']])
    data['ca'] = normalized(ca)
    return data
#########################################################################
class hiddenlayer1 : ##hidden layer 개수 1
    def init_network(): ##hidden layer의 node 수 12
        i = 9
        h = 12
        o = 2
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
        y = sigmoid(a2)
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
        y = sigmoid(a2) ## y를 0과 1 사이의 갑으로 추출.
        if y[0][0] >= y[0][1] : ## y가 0.5 이상이 되면 predict를 1로 선언. 0.5미만의 경우 predict를 0으로 선언.
            predict = 1
        else :
            predict = 0

        if predict == t[0] : ## predict와 target이 같으면 예측 성공, 다르면 예측 실패로 간주.
            error = 0
        else:
            error = 1

        print(predict,t)
        return error
###########################################################################
class hiddenlayer2 : ##hidden layer 개수 2
    def init_network(): ## 첫번째 hidden layer의 node 수 5, 두번째 hidden layer의 node 수 5
        network = {}
        i= 9
        h1 = 5
        h2 = 5
        o = 2
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
        y = sigmoid(a3) ## output layer의 활성홤수는 sigmoid
       
        ##backward
        delta3 = y - t ## error cost = CECF
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
        return temp

    def test(network, x, t):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = sigmoid(a3)
        
        if y[0][0] >= y[0][1] : 
            predict = 1
        else :
            predict = 0

        if predict == t[0] : ## predict와 target이 같으면 예측 성공, 다르면 예측 실패로 간주.
            error = 0
        else:
            error = 1
        return error
#############################################################################

#input
training_data = 240### 8 : 1 : 1 ##303
val_data = 30
test_data = 30
epoch = 200

##network = hiddenlayer1.init_network() ##hidden layer 1개인 경우
network = hiddenlayer2.init_network() ##hidden layer 2개인 경우

df = pd.read_csv('heart.csv', encoding='utf-8')
sdf = setting_data(df) ## for normalizing data

######################### training #######################################
for k in range(epoch):
    for i in range(training_data+val_data):
        x = np.array(sdf.loc[[i],'age':'ca'])
        target = np.array(sdf.loc[[i],'target'])
        if np.sum(target) == 1 :
            t_vector = np.array([1.0,0.0])
        else:
            t_vector = np.array([0.0,1.0])
        ##network = hiddenlayer1.training(network,x,t_vector) ##hidden layer 1개인 경우
        network = hiddenlayer2.training(network,x,t_vector)  ##hidden layer 2개인 경우


####################### test ###############################################
total_error = 0    
for i in range(training_data+val_data,training_data+val_data+test_data): ##test
    x = np.array(sdf.loc[[i],'age':'ca'])
    target = np.array(sdf.loc[[i],'target'])
    if np.sum(target) == 1 :
        t_vector = np.array([1.0,0.0])
    else:
        t_vector = np.array([0.0,1.0])
    
    ##error = hiddenlayer1.test(network,x,t_vector) ##hidden layer 1개인 경우
    error = hiddenlayer2.test(network,x,t_vector)  ##hidden layer 2개인 경우
    total_error += error

error_rate = float(total_error / test_data*100)
print('error = ',error_rate,'%')


