# -*- coding: utf-8 -*- 
#
# Author : hhl <dnrhhl@gmail.com>
#
# Time : 一 10  9 2018 17:34:04
#
#
import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
#print(sigmoid(np.array([0,2])))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim,1))
    return w, b

def propagate(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) 
    cost = np.squeeze(cost)
    dz = A - Y
    dw = (1/m)*np.dot(dz,X.T)
    db = (1/m)*np.sum(dz)
    grads = {'dw':dw,'db':db}
    return grads,cost

#w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
#grads, cost = propagate(w, b, X, Y)
#print(type(cost))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads ,cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i%100 == 0:
            costs.append(cost)
        if print_cost and i % 100 ==0:
            print('cost after iteration '+ str(i) +" " + str(cost))
    params = {'w':w,'b':b}
    grads = {'dw':dw,'db':db}
    return params,grads,costs

#w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
#params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))

def predict(w,b,X):
    m = X.shape[1]
    Y_predition = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if (A[0,i] <= 0.5 ):
            Y_predition[0,i] = 0
        elif(A[0,i] > 0.5):
            Y_predition[0,i] = 1
    return Y_predition

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
print ("predictions = " + str(predict(w, b, X)))
