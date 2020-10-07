import h5py
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored
print('NumPy version', np.__version__)
print('Pandas version', pd.__version__)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 10000)



def print_packages_list():

    """
    Print list of packages used

    :return: None
    """

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    for item in installed_packages_list:
        print(item)
    print('\n')
    return


def write_h5_images(path):

    """
    Write jpg images from h5 format

    :param path: <string> "data" folder path. Must include empty folders "train" and "test"
    :return: None
    """

    hdf_train = h5py.File(path + '/train_catvnoncat.h5', "r")
    hdf_test = h5py.File(path + '/test_catvnoncat.h5', "r")
    train_set_x_orig = np.array(hdf_train["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(hdf_train["train_set_y"][:])
    test_set_x_orig = np.array(hdf_test["test_set_x"][:])
    test_set_y_orig = np.array(hdf_test["test_set_y"][:])
    classes = np.array(hdf_test["list_classes"][:])
    for i in range(len(train_set_x_orig)):
        img = Image.fromarray(train_set_x_orig[i].astype('uint8'), 'RGB')
        img.save(path + 'train/catornot_' + str(i) + '.jpg', "JPEG", subsamplying=0, quality=100)
        # img.save(path + 'train/catornot_down' + str(i) + '.jpg', "JPEG")
    for i in range(len(test_set_x_orig)):
        img = Image.fromarray(test_set_x_orig[i].astype('uint8'), 'RGB')
        img.save(path + 'test/catornot_' + str(i) + '.jpg', "JPEG", subsamplying=0, quality=100)
        # img.save(path + 'test/catornot_down' + str(i) + '.jpg', "JPEG")
    return


def load_dataset(path):

    """
    Load images from h5 file

    :param path: <string> "data" folder path. Must include h5 files"

    :return train_set_x_orig: <numpy ndarray <uint8>> training image data, 3 channel
    :return train_set_y_orig: <numpy ndarray <uint8>> training label data, binary
    :return test_set_x_orig: <numpy ndarray <uint8>> test image data, 3 channel
    :return test_set_y_orig: <numpy ndarray <uint8>> test label data, binary
    :return class: <numpy ndarrray <numpy bytes>> class labels
    """

    train_dataset = h5py.File(path + '/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path +'/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def printm(text, color='white', switch=False):

    """
    Wrapper for built-in print function for inline output color control

    :param text: <string> string to print
    :param color: <string> name of display color
    :param switch: <bool> False to skip colored output. True to invoke
    :return: None
    """

    if switch:
        print(colored(text, color))
    else:
        print(text)
    return


def eda(train, test, filepath, *args):

    """
    Perform some exploratory data analysis

    :param train:
    :param test:
    :param filepath:
    :param args:
    :return:
    """

    printm('Performing exploratory data analysis ... ', color='green', switch=TERMCOLOR)
    # train = train.sample(frac=0.01, replace=True, random_state=42)
    train = train.sample(frac=0.01, replace=True)
    # printm(train.describe())
    # print(test.describe())
    # print(train.head(50))
    train = train.dropna()
    # print(train.isna().sum())
    # print(test.head())
    # print(args[0].head())
    print(train.head())
    print(test.head())

    return


def sigmoid(Z):

    """
    Implement the sigmoid activation in numpy

    :param Z: numpy array of any shape

    :return A: output of sigmoid(z), same shape as Z
    :return cache: returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):

    """
    Implement the RELU function.

    :param Z: Output of the linear layer, of any shape

    :return A: output of relu(z), same shape as Z
    :return cache: returns Z as well, useful during backpropagation
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):

    """
    Implement the backward propagation for a single RELU unit.

    :param dA: post-activation gradient, of any shape
    :param cache: 'Z' where we store for computing backward propagation efficiently

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):

    """
    Implement the backward propagation for a single SIGMOID unit.

    :param dA: post-activation gradient, of any shape
    :param cache: 'Z' where we store for computing backward propagation efficiently

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(n_x, n_h, n_y):

    """
    Initialze weights and biases

    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer

    :return parameters: python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):

    """
    Initialize weights and biases

    :param layer_dims: python array (list) containing the dimensions of each layer in our network

    :return parameters: python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):

    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    :param A: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)

    :return Z: the input of the activation function, also called pre-activation parameter
    :return cache: a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :return A: the output of the activation function, also called the post-activation value
    :return cache:  a python dictionary containing "linear_cache" and "activation_cache";
                    stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    :param X: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep()

    :return AL: last post-activation value
    :return caches: list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):

    """
    Implement the cost function defined by equation (7).

    :param AL: probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    :param cost: cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):

    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    :param dZ: Gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    :return dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return dW: Gradient of the cost with respect to W (current layer l), same shape as W
    :return db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    :param dA: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :return dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return dW: Gradient of the cost with respect to W (current layer l), same shape as W
    :return db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    :param AL: probability vector, output of the forward propagation (L_model_forward())
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches:  list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

    Returns:
    :return grads:   A dictionary with the gradients
                     grads["dA" + str(l)] = ...
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    """
    Update parameters using gradient descent

    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of L_model_backward

    :return parameters:   python dictionary containing your updated parameters
                          parameters["W" + str(l)] = ...
                          parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters):

    """
    This function is used to predict the results of a  L-layer neural network.

    :param X: data set of examples you would like to label
    :param parameters: parameters of the trained model

    :return p: predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    acc = np.sum((p == y) / m)
    print("\tAccuracy: " + str(acc))

    return p


def print_mislabeled_images(classes, X, y, p):

    """
    Plots images where predictions and truth were different.

    :param classes: <numpy array <numpy byte string>> list of labels
    :param X: dataset
    :param y: true labels
    :param p: predictions
    """

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])

    for i in range(num_images):
        index = mislabeled_indices[1][i]
        depth = 3
        subplot_sq_dim = np.ceil(np.sqrt(num_images))
        num_px = int(np.sqrt(np.shape(X[:, index])[0] / depth))
        plt.subplot(subplot_sq_dim, subplot_sq_dim, i + 1)
        plt.imshow(X[:, index].reshape(num_px, num_px, depth), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + \
                  " \n Class: " + classes[y[0, index]].decode("utf-8"), fontsize=6)
        plt.subplots_adjust(hspace=.5)
    plt.show()

    return


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, print_learning_rate=True):

    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    :param X: data, numpy array of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, of length (number of layers + 1).
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps

    :return parameters: parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    convergence_q = []
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("\tCost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    if print_learning_rate==True:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


if __name__ == '__main__':

    print_packages_list()

    DATA_FOLDER = 'data/'
    write_h5_images(DATA_FOLDER)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset(DATA_FOLDER)
    best_params = {"hl_0": 18, "hl_1": 7, "hl_2": 5, "learning_rate": 0.0075}

    # get optimal params
    optimize_best_params = 1
    if optimize_best_params:

        #optimized model
        learning_rate_r = np.linspace(0.0025, 0.01, 10)
        layer_hidden_units = [
                              range(18, 36), # hidden layer 1 possible number of hidden units
                              range(6, 18),  # hidden layer 2 possible number of hidden units
                              [3, 4, 5],     # hidden layer 3 possible number of hidden units
                              ]

        N = 20
        ksplits = 5
        rocs = []
        accuracies = []
        best_auc_avg = 0
        skf = StratifiedKFold(n_splits=ksplits, random_state=0)

        for i in range(N):
            printm('Randomized Search Iteration : ' + str(i+1), color='red', switch=TERMCOLOR)
            hl_0 = random.choice(layer_hidden_units[0])
            hl_1 = random.choice(layer_hidden_units[1])
            hl_2 = random.choice(layer_hidden_units[2])
            learning_rate = random.choice(learning_rate_r)
            for train_index, validation_index in skf.split(train_x_orig, train_y.T):
                X_train, X_validation = train_x_orig[train_index], train_x_orig[validation_index]
                y_train, y_validation = train_y.T[train_index].T, train_y.T[validation_index].T
                X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
                X_validation_flatten = X_validation.reshape(X_validation.shape[0], -1).T
                X_train = X_train_flatten / 255
                X_validation = X_validation_flatten / 255

                layers_dims = [np.shape(X_train)[0], hl_0, hl_1, hl_2, 1]  # 4-layer model
                parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=learning_rate, num_iterations=2400,
                                           print_cost=True, print_learning_rate=False)
                pred_validation = predict(X_validation, y_validation, parameters)
                roc = roc_auc_score(y_validation[0], pred_validation[0].T)
                rocs.append(roc)
                printm('\tAUC_ROC : ' + str(roc), color='yellow', switch=TERMCOLOR)
            roc_avg = np.average(rocs)
            printm('\tAverage AUC_ROC : ' + str(roc_avg), color='green', switch=TERMCOLOR)
            if roc_avg > best_auc_avg:
                best_auc_avg = roc_avg
                best_params["hl_0"] = hl_0
                best_params["hl_1"] = hl_1
                best_params["hl_2"] = hl_2
                best_params["learning_rate"] = learning_rate
        print('\nBest ROC_AUC average after', N,'iterations of KFold Cross Validations:',best_auc_avg)
        print('Best parameters: ', best_params,'\n')

    # use best params now
    printm('Using defined params on the whole training set ...', color='red', switch=TERMCOLOR)
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255
    layers_dims = [np.shape(train_x)[0], best_params["hl_0"], best_params["hl_1"], best_params["hl_2"], 1]  # 4-layer model
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=best_params["learning_rate"], num_iterations=2400, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    roc = roc_auc_score(test_y[0], pred_test[0].T)
    printm('AUC_ROC : ' + str(roc), color='yellow', switch=TERMCOLOR)
    print_mislabeled_images(classes, test_x, test_y, pred_test)