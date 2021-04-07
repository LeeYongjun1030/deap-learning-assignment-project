import numpy as np
import math


   
def perceptron(x ##input values
               ,w ## weight
               ,t ##target values
              ,lr ##learing rate
              ,epoch ## number of epoch
               ):
    for i in range(epoch):
         z = x[0]*w[0]+x[1]*w[1]
         print(' w1 = ',w[0],', w2 = ',w[1],', y = ',z)
         error = z-t
        
         ##update weight
         w[0] -= lr*x[0]*error
         w[1] -= lr*x[1]*error
         
     


class activate_function:
    def sigmoid_function(z):
        return 1.0 / (1.0+ math.exp(-z))

    def softmax(z1,z2, index):
        if index ==0 :
            return math.exp(z1) / (math.exp(z1)+math.exp(z2))
        if index ==1 :
            return math.exp(z2) / (math.exp(z1)+math.exp(z2))
       
    def identity(z):
        return z

    def ReLU(z):
        if z<=0:
            return 0
        else:
            return z

    
def feed_forward(num_input ##number of input nodes
                 ,num_hidden ##number of hidden nodes
                 ,num_output ##number of output nodes
                 ,x ## input value
                 ,w_ij,b_j, ##weight and bias
                 w_jk,b_k
                 ,activate_option_in_output_layer
                 ):
    hidden_z = [] ## z value in hidden layer
    hidden_a = [] ## a value in hidden layer
    output_z = [] ## z value in output layer
    output_a = [] ## a value in output layer

    for i in range(num_hidden):
        hidden_z.append( x[0]*w_ij[i] + x[1]*w_ij[i+3] +b_j[i] ) 
        hidden_a.append(activate_function.sigmoid_function(hidden_z[i]))
       
    for i in range(num_output):
        output_z.append((hidden_a[0]*w_jk[i]) + (hidden_a[1]*w_jk[i+2]) + (hidden_a[2]*w_jk[i+4]) + b_k[i])
     
    if activate_option_in_output_layer == 0 : ##activate function = softmax
        output_a.append(activate_function.softmax(output_z[0],output_z[1],0))
        output_a.append(activate_function.softmax(output_z[0],output_z[1],1))

    if activate_option_in_output_layer == 1 : ##activate function = identity
        output_a.append(activate_function.identity(output_z[0]))
        output_a.append(activate_function.identity(output_z[1]))


    return hidden_z,hidden_a,output_z,output_a

    
def backpropagation(x
                    ,w_ij,b_j ##weight and bias
                    ,w_jk,b_k ##weight and bias
                    ,hidden_a ## a value in hidden layer
                    ,output_a  ## a value in output layer
                    ,target ##target value
                    ,lr ## learing rate
                    ,error_function_option ## index 0 is SSE, index 1 is Logistic
                    ):

    delta_j = [0,0,0] ## init delta value in hidden layer
    delta_k = [0,0] ## init delta value in output layer

    if error_function_option == 0 : ## SSE and Softmax
        delta_k[0] = (output_a[0] - target[0])*output_a[0]*(1-output_a[0])+(output_a[1]-target[1])*(-1*output_a[1]*output_a[0])
        delta_k[1] = (output_a[0] - target[0])*(-1*output_a[1]*output_a[0])+(output_a[1]-target[1])*output_a[1]*(1-output_a[1])
    
    if error_function_option == 1 : ## Logistic Error and Softmax  ## identity and SSE
        delta_k[0] = output_a[0] - target[0]
        delta_k[1] = output_a[1] - target[1]
    
  ## sigmoid
    delta_j[0] = (w_jk[0]*delta_k[0]+w_jk[1]*delta_k[1])*hidden_a[0]*(1-hidden_a[0])
    delta_j[1] = (w_jk[2]*delta_k[0]+w_jk[3]*delta_k[1])*hidden_a[1]*(1-hidden_a[1])
    delta_j[2] = (w_jk[4]*delta_k[0]+w_jk[5]*delta_k[1])*hidden_a[2]*(1-hidden_a[2])

    ## update weight and bias in output layer  
    w_jk[0] -= lr*hidden_a[0]*delta_k[0] ## j = 2 , k = 5 
    w_jk[1] -= lr*hidden_a[0]*delta_k[1] ## j = 2 , k = 6 
    w_jk[2] -= lr*hidden_a[1]*delta_k[0] ## j = 3 , k = 5 
    w_jk[3] -= lr*hidden_a[1]*delta_k[1] ## j = 3 , k = 6 
    w_jk[4] -= lr*hidden_a[2]*delta_k[0] ## j = 4 , k = 5 
    w_jk[5] -= lr*hidden_a[2]*delta_k[1] ## j = 4 , k = 6 
    b_k[0] -= lr*delta_k[0] ## k = 5
    b_k[1] -= lr*delta_k[1] ## k = 6

    ## update weight and bias in hidden layer  
    w_ij[0] -= lr*x[0]*delta_j[0] ## i = 0 , j = 2 
    w_ij[1] -= lr*x[0]*delta_j[1] ## i = 0 , j = 3 
    w_ij[2] -= lr*x[0]*delta_j[2] ## i = 0 , j = 4 
    w_ij[3] -= lr*x[1]*delta_j[0] ## i = 1 , j = 2 
    w_ij[4] -= lr*x[1]*delta_j[1] ## i = 1 , j = 3 
    w_ij[5] -= lr*x[1]*delta_j[2] ## i = 1 , j = 4
    b_j[0] -= lr*delta_j[0] ## j = 2
    b_j[1] -= lr*delta_j[1] ## j = 3
    b_j[2] -= lr*delta_j[2] ## j = 4
  
    return w_ij,b_j,w_jk,b_k



    
def feed_forward_2_hidden_layer(x ## input value
                                ,w_ij,b_j ##weight and bias
                                ,w_jk,b_k
                                ,w_kl,b_l                            
                                ):
    hidden_1_z = [] ## z value in hidden 1 layer
    hidden_1_a = [] ## a value in hidden 1 layer
    hidden_2_z = [] ## z value in hidden 2 layer
    hidden_2_a = [] ## a value in hidden 2 layer
    output_z = [] ## z value in output layer
    output_a = [] ## a value in output layer

    for i in range(3):
        hidden_1_z.append( x[0]*w_ij[i] + x[1]*w_ij[i+3] +b_j[i] ) 
        hidden_1_a.append(activate_function.sigmoid_function(hidden_1_z[i]))

    for i in range(2):
        hidden_2_z.append( hidden_1_z[0]*w_jk[i] + hidden_1_z[1]*w_jk[i+2] + hidden_1_z[2]*w_jk[i+4]  +b_k[i] ) 
        hidden_2_a.append(activate_function.sigmoid_function(hidden_2_z[i]))
       
    for i in range(2):
        output_z.append((hidden_2_a[0]*w_kl[i]) + (hidden_2_a[1]*w_kl[i+2])+ b_l[i])
     
    ##activate function = softmax
    output_a.append(activate_function.softmax(output_z[0],output_z[1],0))
    output_a.append(activate_function.softmax(output_z[0],output_z[1],1))

    return hidden_1_z,hidden_1_a,hidden_2_z,hidden_2_a,output_z,output_a

def backpropagation2(x
                    ,w_ij,b_j ##weight and bias
                    ,w_jk,b_k ##weight and bias
                    ,w_kl,b_l ##weight and bias
                    ,hidden_1_a ## a value in hidden 1 layer
                    ,hidden_2_a ## a value in hidden 2 layer
                    ,output_a  ## a value in output layer
                    ,target ##target value
                    ,lr ## learing rate
                    ,error_function_option ## index 0 is SSE, index 1 is Logistic error
                    ):

    delta_j = [0,0,0] ## init delta value in hidden 1 layer
    delta_k = [0,0] ## init delta value in hidden 2 layer
    delta_l = [0,0] ## init delta value in output layer

    if error_function_option == 0 : ## SSE and Softmax
        delta_l[0] = (output_a[0] - target[0])*output_a[0]*(1-output_a[0])+(output_a[1]-target[1])*(-1*output_a[1]*output_a[0])
        delta_l[1] = (output_a[0] - target[0])*(-1*output_a[1]*output_a[0])+(output_a[1]-target[1])*output_a[1]*(1-output_a[1])
    
    if error_function_option == 1 : ## Logistic Error and Softmax  ## identity and SSE
        delta_l[0] = output_a[0] - target[0]
        delta_l[1] = output_a[1] - target[1]
    
    delta_k[0] = (w_kl[0]*delta_l[0]+w_kl[1]*delta_l[1])*hidden_2_a[0]*(1-hidden_2_a[0])
    delta_k[1] = (w_kl[2]*delta_l[0]+w_kl[3]*delta_l[1])*hidden_2_a[1]*(1-hidden_2_a[1])

    delta_j[0] = (w_jk[0]*delta_k[0]+w_jk[1]*delta_k[1])*hidden_1_a[0]*(1-hidden_1_a[0])
    delta_j[1] = (w_jk[2]*delta_k[0]+w_jk[3]*delta_k[1])*hidden_1_a[1]*(1-hidden_1_a[1])
    delta_j[2] = (w_jk[4]*delta_k[0]+w_jk[5]*delta_k[1])*hidden_1_a[2]*(1-hidden_1_a[2])

    ## update weight and bias in output layer  
    w_kl[0] -= lr*hidden_2_a[0]*delta_l[0] ## k = 5 , l = 7 
    w_kl[1] -= lr*hidden_2_a[0]*delta_l[1] ## k = 5 , l = 8 
    w_kl[2] -= lr*hidden_2_a[1]*delta_l[0] ## k = 6 , l = 7 
    w_kl[3] -= lr*hidden_2_a[1]*delta_l[1] ## k = 6 , l = 8 
    b_l[0] -= lr*delta_l[0] ## l = 7
    b_l[1] -= lr*delta_l[1] ## l = 8

    ## update weight and bias in hidden 2 layer  
    w_jk[0] -= lr*hidden_1_a[0]*delta_k[0] ## j = 2 , k = 5 
    w_jk[1] -= lr*hidden_1_a[0]*delta_k[1] ## j = 2 , k = 6 
    w_jk[2] -= lr*hidden_1_a[1]*delta_k[0] ## j = 3 , k = 5 
    w_jk[3] -= lr*hidden_1_a[1]*delta_k[1] ## j = 3 , k = 6 
    w_jk[4] -= lr*hidden_1_a[2]*delta_k[0] ## j = 4 , k = 5 
    w_jk[5] -= lr*hidden_1_a[2]*delta_k[1] ## j = 4 , k = 6 
    b_k[0] -= lr*delta_k[0] ## k = 5
    b_k[1] -= lr*delta_k[1] ## k = 6

    ## update weight and bias in hidden 1 layer  
    w_ij[0] -= lr*x[0]*delta_j[0] ## i = 0 , j = 2 
    w_ij[1] -= lr*x[0]*delta_j[1] ## i = 0 , j = 3 
    w_ij[2] -= lr*x[0]*delta_j[2] ## i = 0 , j = 4 
    w_ij[3] -= lr*x[1]*delta_j[0] ## i = 1 , j = 2 
    w_ij[4] -= lr*x[1]*delta_j[1] ## i = 1 , j = 3 
    w_ij[5] -= lr*x[1]*delta_j[2] ## i = 1 , j = 4
    b_j[0] -= lr*delta_j[0] ## j = 2
    b_j[1] -= lr*delta_j[1] ## j = 3
    b_j[2] -= lr*delta_j[2] ## j = 4
  
    return w_ij,b_j,w_jk,b_k,w_kl,b_l
    
######################## Problem 1 #######################################################
x = [1.5, -2.0] 
w = [0.7, 0.5]
t = 1.0
lr = 0.1
epoch= 10
print(' Problem 1 ')
perceptron(x,w,t,lr,epoch)

######################## Problem 2 #######################################################
num_input = 2  ##number of input nodes
num_hidden = 3 ##number of hidden nodes
num_output = 2 ##number of output nodes

x = np.array([1.0,1.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from hidden layer to output layer
b_k = [-0.1,-0.2]

t = [0.0, 1.0] ## target
lr = 0.2 ##learning rate
epoch = 100

hidden_z = [] ## z value in hidden layer
hidden_a = [] ## a value in hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 2 ')
for i in range(0,epoch):
    hidden_z,hidden_a,output_z,output_a= feed_forward(num_input, num_hidden, num_output,x,w_ij,b_j,w_jk,b_k,0)
    print(output_a)
    w_ij,b_j,w_jk,b_k = backpropagation(x,w_ij,b_j,w_jk,b_k,hidden_a,output_a,t,lr,0)
   
    
######################## Problem 3 #######################################################
num_input = 2  ##number of input nodes
num_hidden = 3 ##number of hidden nodes
num_output = 2 ##number of output nodes

x = np.array([10.0,5.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from hidden layer to output layer
b_k = [-0.1,-0.2]

t = [8.0, 40.0] ## target
lr = 0.2 ##learning rate
epoch = 30

hidden_z = [] ## z value in hidden layer
hidden_a = [] ## a value in hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 3 ')
for i in range(0,epoch):
    hidden_z,hidden_a,output_z,output_a= feed_forward(num_input, num_hidden, num_output,x,w_ij,b_j,w_jk,b_k,1)
    print(output_a)
    w_ij,b_j,w_jk,b_k = backpropagation(x,w_ij,b_j,w_jk,b_k,hidden_a,output_a,t,lr,1)
   
############################## Problem 4 #################################################
num_input = 2  ##number of input nodes
num_hidden = 3 ##number of hidden nodes
num_output = 2 ##number of output nodes

x = np.array([1.0,1.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from hidden layer to output layer
b_k = [-0.1,-0.2]

t = [0.0, 1.0] ## target
lr = 0.2 ##learning rate
epoch = 100

hidden_z = [] ## z value in hidden layer
hidden_a = [] ## a value in hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 4 ')
for i in range(0,epoch):
    hidden_z,hidden_a,output_z,output_a= feed_forward(num_input, num_hidden, num_output,x,w_ij,b_j,w_jk,b_k,0)
    print(output_a)
    w_ij,b_j,w_jk,b_k = backpropagation(x,w_ij,b_j,w_jk,b_k,hidden_a,output_a,t,lr,0)

  ################################ Problem 5 ##############################################

x = np.array([1.0,1.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from first hidden layer to second hidden layer
b_k = [-0.1,-0.2]

w_kl = [0.3,0.1,0.2,0.4] ##from second hidden layer to output layer
b_l = [-0.2,-0.1]

t = [0.0, 1.0] ## target
lr = 0.2 ##learning rate
epoch = 150

hidden_1_z = [] ## z value in first hidden layer
hidden_1_a = [] ## a value in first hidden layer
hidden_2_z = [] ## z value in second hidden layer
hidden_2_a = [] ## a value in second hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 5 ')
for i in range(0,epoch):
    hidden_1_z,hidden_1_a,hidden_2_z,hidden_2_a,output_z,output_a = feed_forward_2_hidden_layer(x,w_ij,b_j,w_jk,b_k,w_kl,b_l)
    print(output_a)
    w_ij,b_j,w_jk,b_k,w_kl,b_l = backpropagation2(x,w_ij,b_j,w_jk,b_k,w_kl,b_l,hidden_1_a,hidden_2_a,output_a,t,lr,0)




  ############################## Problem 6 #################################################3
num_input = 2  ##number of input nodes
num_hidden = 3 ##number of hidden nodes
num_output = 2 ##number of output nodes

x = np.array([1.0,1.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from hidden layer to output layer
b_k = [-0.1,-0.2]

t = [0.0, 1.0] ## target
lr = 0.2 ##learning rate
epoch = 30

hidden_z = [] ## z value in hidden layer
hidden_a = [] ## a value in hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 6 ')
for i in range(0,epoch):
    hidden_z,hidden_a,output_z,output_a= feed_forward(num_input, num_hidden, num_output,x,w_ij,b_j,w_jk,b_k,0)
    print(output_a)
    w_ij,b_j,w_jk,b_k = backpropagation(x,w_ij,b_j,w_jk,b_k,hidden_a,output_a,t,lr,1)


############################### Problem 7 #################################################
x = np.array([1.0,1.0]) ## input value 

w_ij = [0.1,0.2,0.1,0.2,0.1,0.2] ##from input layer to hidden layer
b_j = [-0.1,-0.2,-0.1]

w_jk = [0.2,0.1,0.1,0.2,0.2,0.1] ##from first hidden layer to second hidden layer
b_k = [-0.1,-0.2]

w_kl = [0.3,0.1,0.2,0.4] ##from second hidden layer to output layer
b_l = [-0.2,-0.1]

t = [0.0, 1.0] ## target
lr = 0.2 ##learning rate
epoch = 50

hidden_1_z = [] ## z value in first hidden layer
hidden_1_a = [] ## a value in first hidden layer
hidden_2_z = [] ## z value in second hidden layer
hidden_2_a = [] ## a value in second hidden layer
output_z = [] ## z value in output layer
output_a = [] ## a value in output layer

print(' ')
print(' Problem 7 ')
for i in range(0,epoch):
    hidden_1_z,hidden_1_a,hidden_2_z,hidden_2_a,output_z,output_a = feed_forward_2_hidden_layer(x,w_ij,b_j,w_jk,b_k,w_kl,b_l)
    print(output_a)
    w_ij,b_j,w_jk,b_k,w_kl,b_l = backpropagation2(x,w_ij,b_j,w_jk,b_k,w_kl,b_l,hidden_1_a,hidden_2_a,output_a,t,lr,1)

