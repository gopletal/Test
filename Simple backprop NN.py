# NEURAL NETWORK FOUNDATIONS 
#
# These examples are taken and adapted from the blog, 
# http://iamtrask.github.io/2015/07/12/basic-python-network/



#%%
#
# SIMPLE NN FORWARD PROPAGATION
# (3 inputs nodes(l0),1 ouput node(l1)
# 4 training sets, full batch training)
#
#  l0(input)      w0(weight1)      l1(Output)
#     l01           l01-l11
#                     
#     l02           l02_l11           l11   
#                     
#     l03           l03-l11
#
#

import numpy as np

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))    

#Sigmoid derivative (could be sped up dx(sigmoid)=sigmoid*(1-sigmoid))
def sigmoid_dervative(x):
    return (1/(1+np.exp(-x))) * (1-(1/(1+np.exp(-x))))


#input dataset 
data_in  = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output datset 
data_out = np.array([[0],[0],[1],[1]]) 
 
#seed radom numbers 
np.random.seed(1)

#initialize weights randomly between -1 and 1 with mean 0
w0 = 2*np.random.random((3,1)) - 1     #3 l0 nodes to 1 l1 node

for loop in range(10000):

    #forward propagation
    l0 = data_in
    l1 = sigmoid(np.dot(l0,w0))

    #difference between last layer and data (4 X 1 matrix)
    l1_error = data_out - l1

    #multiply difference by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid_dervative(np.dot(l0,w0)) 

    #update weights
    w0 += np.dot(l0.T,l1_delta)

    #write out error   
    if(loop% 1000) == 0:
       print("Step: "+str(loop).zfill(4)+ "   Error: "+str(np.mean(np.abs(l1_error)))) 


print ("\n"+"Output after training")
print (l1)





#%%
#
# NN WITH BACKPROPAGATION (1 HIDDEN LAYER)
# (3 inputs nodes(l0),4 hidden nodes(l1),1 ouput node(l2)
# 4 training sets, full batch training)
# 
#
#   l0(input)            w0(weight1)                l1(Hidden1)     w1(weight2)    l2(Output)
#     l01      l01-l11,l01-l12,l01-l13,l01-l14          l11           l11-l21       
#                            
#     l02      l02-l11,l02-l12,l02-l13,l02-l14          l12           l12-l21          l21
#                     
#     l03      l03-l11,l03-l12,l03-l13,l03,l14          l13           l13-l21  
#                   
#                                                       l14           l14-l21   
#
#


#input dataset 
data_in  = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output datset 
data_out = np.array([[0],[1],[1],[0]]) 
 
#seed radom numbers 
np.random.seed(1)

#initialize weights randomly between -1 and 1 with mean 0  
w0 = 2*np.random.random((3,4)) - 1    #3 l0 nodes to 4 l1 nodes (12 weights)
w1 = 2*np.random.random((4,1)) - 1    #4 l1 nodes to 1 l2 node (4 weights) 

for loop in range(10000):

    #forward propagation
    l0 = data_in
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))

    #difference between last layer and data (4 X 1 matrix)
    l2_error = data_out - l2

    #multiply difference by the slope of the sigmoid at the values in l1
    l2_delta = l2_error * sigmoid_dervative(np.dot(l1,w1)) 

    #Backpropagate to workout l1 error contribution to l2
    l1_error = np.dot(l2_delta,w1.T)
    l1_delta = l1_error * sigmoid_dervative(np.dot(l0,w0)) 

    #update weights
    w1 += np.dot(l1.T,l2_delta)
    w0 += np.dot(l0.T,l1_delta)

    #write out error   
    if(loop% 100) == 0:
       print("Step: "+str(loop).zfill(4)+ "   Error: "+str(np.mean(np.abs(l1_error)))) 


print ("\n"+"Output after training")
print (l2)



#%%
