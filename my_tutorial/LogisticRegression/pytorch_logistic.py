from data_core import load_dataset, data_transform_for_numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable


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


class Logstic_Regression(nn.Module):
	def __init__(self, in_dim, n_class):
		super(Logstic_Regression, self).__init__()
		self.logstic = nn.Linear(in_dim, n_class)
	
	def forward(self, x):
		out = self.logstic(x)
		return out


learning_rate = 1e-3
num_epoches = 500
model = Logstic_Regression(nx, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
	X = Variable(torch.from_numpy(train_set_x).float())
	Y = Variable(torch.from_numpy(train_set_y).long())
	out = model(X)
	loss = criterion(out, Y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if epoch % 100 == 0:
		print(loss)
