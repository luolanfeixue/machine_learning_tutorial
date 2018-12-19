from data_core import load_dataset,data_transform_for_numpy
import numpy as np
import matplotlib.pyplot as plt

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

# 展示样本例子
is_show_example = False
show_example(is_show_example)
# 展示数据shape
is_show_shape = False
show_shape(is_show_shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict_batch(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0][i] >= 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def train(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # 定义参数
    nx = X_train.shape[0]
    m = X_train.shape[1]
    w = np.zeros((nx, 1))
    b = 0
    dw = np.zeros(w.shape)
    db = 0
    costs = []
    for i in range(num_iterations):
        A = sigmoid(np.dot(w.T, X_train) + b)
        A = A.reshape(A.shape[0], -1)
        cost = -np.sum(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))/m
        dz = A - Y_train
        dw = (1/m)*np.dot(X_train,dz.T)
        db = (1/m)*np.sum(dz)
        cost = np.squeeze(cost)

        assert (dw.shape == w.shape)

        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    Y_prediction_test = predict_batch(w, b, X_test)
    Y_prediction_train = predict_batch(w, b, X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = train(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = train(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()