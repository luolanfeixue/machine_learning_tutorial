from data_core import load_dataset,data_transform_for_numpy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop




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
train_set_x, train_set_y, test_set_x, test_set_y = train_set_x.T, train_set_y.T, test_set_x.T, test_set_y.T


m = train_set_x.shape[0]
nx = train_set_x.shape[1]



model = Sequential([
    Dense(1, input_dim=nx),
    Activation('sigmoid')
])


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
print('Training ------------')
model.fit(train_set_x, train_set_y, epochs=10, batch_size=32) #keras_v1 epochs 2 nb_epoch


print('Testing ------------')
loss, accuracy = model.evaluate(test_set_x, test_set_y)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

