# -*- coding: utf-8 -*-
#
# Author : hhl <dnrhhl@gmail.com>
#
# Time : 一 10  9 2018 14:38:29
#
#
import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

x = np.array([1,2,3])
s = sigmoid(x)
t = sigmoid_derivative(x)
print('sigmoid',s)
print('sigmoid_derivative',t)

# axis = 0 表示跨行操作，axia=1 表示跨列操作。
# 对每行求norm，所以要将每一行对所有列都加起来。 这是跨列操作，所以axis=1
# x shape (2,3) x_norm shape (2,1) 这里会将(2,1) 复制3份
# np.linalg.norm() 求norm的函数，可以求L1，也可以求L2
def normalize(x):
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    x = x/x_norm
    return x

x = [0,3,4,2,6,4]
x = np.array(x).reshape(2,3)
x = normalize(x)
print('normalize',x)


#矩阵和向量通用softmax
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s 

x = np.array([
    [9,2,5,0,0],
    [7,5,0,0,0]
    ])
x = softmax(x)
print('softmax',x)

def L1(y_hat,y):
    loss = np.sum(np.abs(y_hat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))


def L2(y_hat, y):
    loss = np.sum(np.multiply(y -  y_hat, y - y_hat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
