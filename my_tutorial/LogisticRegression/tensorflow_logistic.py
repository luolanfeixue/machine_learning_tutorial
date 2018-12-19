from data_core import load_dataset,data_transform_for_numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def show_example(is_show_example):
    if is_show_example:
        index = 25
        plt.imshow(train_set_x_orig[index])
        plt.show()
        print("y = " + str(train_set_y[:, index]) + ", it's a '" + \
              classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

def show_shape(is_show_shape):
    if is_show_shape:
        print('train_set_x.shape', train_set_x.shape)
        print('train_set_y.shape', train_set_y.shape)
        print('test_set_x.shape', test_set_x.shape)
        print('test_set_y.shape', test_set_y.shape)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x, test_set_x = data_transform_for_numpy(train_set_x_orig, test_set_x_orig)

train_set_x,train_set_y,test_set_x, test_set_y  = train_set_x.T,train_set_y.T,test_set_x.T, test_set_y.T
# show_shape(True)


m = train_set_x.shape[0]
nx = train_set_x.shape[1]


X = tf.placeholder(tf.float32, [None, nx]) # 图像数据 nx
Y = tf.placeholder(tf.float32, [None, 1]) # 图像类别


w1 = tf.Variable(tf.zeros([nx, 1]))
b1 = tf.Variable(tf.zeros([1]))
z1 = tf.matmul(X, w1) + b1
a1 = tf.nn.sigmoid(z1)
#
# w2 = tf.Variable(tf.zeros([10, 1]))
# b2 = tf.Variable(tf.zeros([1]))
# z2 = tf.matmul(a1, w2) + b2
# a2 = tf.nn.sigmoid(z2)

# 损失函数用cross entropy
learning_rate = 0.005
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z1,labels=Y)) ## z1要用没求logistic 之前的tensor
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

y_prediction = tf.squeeze(tf.round(a1))
correct_prediction = tf.equal(y_prediction, Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2000):
        _, c = sess.run([optimizer, cost], feed_dict={X:train_set_x, Y:train_set_y})
        acc_value_test = sess.run(accuracy, feed_dict={X: test_set_x, Y: test_set_y})
        acc_value_train = sess.run(accuracy, feed_dict={X: train_set_x, Y: train_set_y})
        if epoch % 100 == 0:
            print("Epoch:", '%d' % epoch, "cost=", "{:.6f}".format(c))
            print('Accuracy on train set: ' + str(acc_value_train), 'Accuracy on test set: ' + str(acc_value_test))