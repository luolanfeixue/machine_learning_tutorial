# -*- coding: utf-8 -*- 
# # Author : hhl <dnrhhl@gmail.com>
#
# Time : 日 30  9 2018 18:16:19
#
#
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


y_hat = tf.constant(36,name='y_hat')
y = tf.constant(39,name='y')
loss = tf.Variable((y-y_hat)**2,name='loss')
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print('variable:',session.run(loss))

print()

a = tf.constant(2)
b = tf.constant(4)
c = tf.multiply(a,b)
with tf.Session() as sess:
    print ('multiply:',sess.run(c))

print ()

x = tf.placeholder(tf.int64, name='x')
with tf.Session() as sess:
    r = sess.run(x*2, feed_dict={x:3})
    print ('placeholders:',r)

print ()

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = 'X')
    W = tf.constant(np.random.randn(4,3), name = 'W')
    b = tf.constant(np.random.randn(4,1), name = 'b')

    Y = tf.matmul(W, X) + b

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    return result

print ("linear function = " + str(linear_function()))

print ()


def sigmoid(z):
    x = tf.placeholder(tf.float32, name = 'x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict = {x:z})
    return result
print ('sigomid')
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

print ()

def cost(logits, labels):
    z = tf.placeholder(tf.float32, name = 'logits')
    y = tf.placeholder(tf.float32, name = 'labels')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)

    sess = tf.Session()
    cost = sess.run(cost, feed_dict = {z:logits, y:labels}) 
    sess.close()

    return cost

print ('cost:')
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))

print ()
    
def one_hot_matrix(labels,C):
    C = tf.constant(C, name = 'C')
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


print ('one_hot')
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))
print ()

print ("ones:")
def ones(shape):
    ones = tf.ones(shape)

    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return ones

print ("ones = " + str(ones([3,3])))

print ()

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ('DATA:')
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

print ()

def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape = (n_x, None), dtype = tf.float32)
    Y = tf.placeholder(shape = (n_y, None), dtype = tf.float32)

    return X, Y

print ('placeholder:')
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
print ()

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

print('initialize:')
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

print()

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2   
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3,A2) + b3
    return Z3

print('forward_propagation:')
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))

print()

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

print('compute_cost:')
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int( m / minibatch_size )
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches :
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
