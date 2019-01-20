import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) # train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:]) # train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:]) # test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:]) # test set labels

    classes = np.array(test_dataset['list_classes'][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
 

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def data_transform_for_numpy(train_set_x_orig, test_set_x_orig):
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]

    train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    return train_set_x, test_set_x

