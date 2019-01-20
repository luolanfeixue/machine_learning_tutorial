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
learning_rate = 0.01

m = train_set_x.shape[0]
nx = train_set_x.shape[1]


x = tf.placeholder(tf.float32, [None, nx]) # 图像数据 nx
y = tf.placeholder(tf.float32, [None, 1]) # 图像类别
W = tf.Variable(tf.zeros([nx, 1]))
b = tf.Variable(tf.zeros([1]))
pred = tf.nn.sigmoid(tf.matmul(x, W) + b)
correct_prediction = tf.equal(tf.gs(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 损失函数用cross entropy
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    _, cost = sess.run([optimizer, cost], feed_dict={x:train_set_x, y:train_set_y})
    print("Training Accuracy:", accuracy.eval({x: train_set_x, y: train_set_y}))
    print("Testing Accuracy:", accuracy.eval({x: test_set_x, y: test_set_y}))