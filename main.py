import h5py
import math
import time
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
from PIL import Image
from skimage import transform, io
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


def load_ext_images(data_path):

    """
    Load jpg or jpeg images into ndarray

    :param data_path: <string> path to images directory
    :return all_images: <ndarray <float?>> image data
    """

    all_images = []
    for file in glob.glob(data_path + '/*.jpg'):
        image = np.array(io.imread(file))
        all_images.append(image)
    for file in glob.glob(data_path + '/*.jpeg'):
        image = np.array(io.imread(file))
        all_images.append(image)

    return all_images


def resize(images_resize, dims):

    """
    Resize images

    :param images_resize: <ndarray <float>> image data to transform
    :param dims: <tuple <int>> tuple of width and height of new picture
    :return: <ndarray <float>> image data resized
    """

    width = dims[0]
    height = dims[1]
    res_img = []
    for im in images_resize:
        my_image = transform.resize(im, (width, height), preserve_range=True, anti_aliasing=True)
        res_img.append(my_image)

    return res_img


def image_preprocess(all_images):

    """
    Preprocess images from external sourc, e.g Google images

    :param all_images: <ndarray <float>> image data to preprocess
    :return images_processed: <ndarray <float>> preprocessed images in compatible form for model training
    """

    images_processed = resize(all_images, (64, 64))

    return images_processed


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
    :return test_classes: <numpy ndarrray <numpy bytes>> class labels
    """

    train_dataset = h5py.File(path + '/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path +'/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    test_classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, test_classes


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


def compute_cost_minibatch(AL, Y):

    """
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function without dividing by number of training examples

    Note:
    This is used with mini-batches,
    so we'll first accumulate costs over an entire epoch
    and then divide by the m training examples
    """

    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost_total =  np.sum(logprobs)

    return cost_total


def compute_cost_minibatch_L2_regularization(AL, Y, parameters, _lambda):

    """
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function without dividing by number of training examples

    Note:
    This is used with mini-batches,
    so we'll first accumulate costs over an entire epoch
    and then divide by the m training examples
    """
    l2_regularization_cost = np.sum([np.sum(np.square(v)) for k, v in parameters.items() if 'W' in k]) * _lambda / 2
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost_total =  np.sum(logprobs) + l2_regularization_cost
    cost_total = np.squeeze(cost_total)
    assert (cost_total.shape == ())

    return cost_total


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


def linear_backward_regularization(dZ, cache, _lambda):

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

    dW = 1. / m * np.dot(dZ, A_prev.T) + _lambda / m * W
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


def linear_activation_backward_regularization(dA, cache, _lambda, activation):

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
        dA_prev, dW, db = linear_backward_regularization(dZ, linear_cache, _lambda)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_regularization(dZ, linear_cache, _lambda)

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


def L_model_backward_regularization(AL, Y, caches, _lambda):

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
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_regularization(dAL, current_cache, _lambda,
                                                                                                  activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_regularization(grads["dA" + str(l + 2)], current_cache, _lambda,
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


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
        ### END CODE HERE ###

    return parameters, v


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l + 1)] = parameters['W' + str(l + 1)] * 0
        v["db" + str(l + 1)] = parameters['b' + str(l + 1)] * 0
        ### END CODE HERE ###

    return v


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k + 1)]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * (num_complete_minibatches) - 1: -1]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (num_complete_minibatches) - 1: -1]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


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
        # parameters = update_parameters(parameters, grads, learning_rate)
        parameters = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)

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


def L_layer_model_minibatch_momentum(X, Y, layers_dims, learning_rate=0.0007, mini_batch_size=64,
                                     beta=0.9, num_epochs=10000, print_cost=True):

    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    :param X: data, numpy array of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, of length (number of layers + 1).
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps
    :param num_epochs: number of complete passes through the training set

    :return parameters: parameters learnt by the model. They can then be used to predict.
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims)
    v = initialize_velocity(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost_minibatch(AL, minibatch_Y)

            # Backward propagation
            # grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            grads = L_model_backward(AL, minibatch_Y, caches)

            # Update parameters
            parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)

        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def L_layer_model_minibatch_momentum_regularization(X, Y, layers_dims, learning_rate=0.0007, mini_batch_size=64,
                                     beta=0.9, num_epochs=10000, print_cost=True, _lambda=0.1, plot_costs=True):

    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    :param X: data, numpy array of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, of length (number of layers + 1).
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps
    :param num_epochs: number of complete passes through the training set
    :param mini_batch_size:
    :param beta:
    :param _lambda: regularization parameter. Range [0,1]. 0 eliminates the effects
    :param plot_costs: False to suppress plotting learning rate curve. False to avoid user interaction

    :return parameters: parameters learnt by the model. They can then be used to predict.
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims)
    v = initialize_velocity(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost_minibatch_L2_regularization(AL, minibatch_Y, parameters, _lambda)

            # Backward propagation
            # grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            grads = L_model_backward_regularization(AL, minibatch_Y, caches, _lambda)

            # Update parameters
            parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)

        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    if plot_costs:
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters


if __name__ == '__main__':

    a = time.time()
    print_packages_list()

    DATA_FOLDER = 'data/'
    write_h5_images(DATA_FOLDER)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset(DATA_FOLDER)

    # load and preprocess data from external sources to add to the training set
    DATA_FOLDER = 'data/cat_google_search/'
    images = load_ext_images(DATA_FOLDER)
    images = image_preprocess(images)
    labels = np.ones((1, len(images)))
    train_x_orig = np.concatenate((train_x_orig, images))
    train_y = np.concatenate((train_y, labels), axis=1)
    DATA_FOLDER = 'data/notcat_google_search/'
    images = load_ext_images(DATA_FOLDER)
    images = image_preprocess(images)
    labels = np.zeros((1, len(images)))
    train_x_orig = np.concatenate((train_x_orig, images))
    train_y = np.concatenate((train_y, labels), axis=1)

    best_params = {"hl_0": 24, "hl_1": 12, "hl_2": 5, "learning_rate": 0.00075,
                   "mini_batch_size": 32, "beta": 0.9, "_lambda": 0.1}

    # get optimal params
    optimize_best_params = 1
    if optimize_best_params:

        #optimized model
        learning_rate_range = np.linspace(0.0025, 0.01, 10)
        layer_hidden_units = [
                              range(18, 36), # hidden layer 1 possible number of hidden units
                              range(6, 18),  # hidden layer 2 possible number of hidden units
                              [3, 4, 5],     # hidden layer 3 possible number of hidden units
                              ]
        mini_batch_size_range = np.logspace(2.0, 6.0, num=5, base=2)
        beta_range = 1 - np.logspace(1, 3, num=3, base=0.1)
        lambda_range = np.linspace(0.0, 1, 10)

        N = 20
        ksplits = 5
        rocs = []
        accuracies = []
        best_auc_avg = 0
        skf = StratifiedKFold(n_splits=ksplits, random_state=0)

        for i in range(N):
            printm('\nRandomized Search Iteration : ' + str(i+1), color='red', switch=TERMCOLOR)
            hl_0 = random.choice(layer_hidden_units[0])
            hl_1 = random.choice(layer_hidden_units[1])
            hl_2 = random.choice(layer_hidden_units[2])
            learning_rate = random.choice(learning_rate_range)
            mini_batch_size = int(random.choice(mini_batch_size_range))
            beta = random.choice(beta_range)
            _lambda = random.choice(lambda_range)
            fold = 0

            for train_index, validation_index in skf.split(train_x_orig, train_y.T):
                fold += 1
                X_train, X_validation = train_x_orig[train_index], train_x_orig[validation_index]
                y_train, y_validation = train_y.T[train_index].T, train_y.T[validation_index].T
                X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
                X_validation_flatten = X_validation.reshape(X_validation.shape[0], -1).T
                X_train = X_train_flatten / 255
                X_validation = X_validation_flatten / 255

                layers_dims = [np.shape(X_train)[0], hl_0, hl_1, hl_2, 1]

                # # base model
                # parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=learning_rate,
                #                            num_iterations=2400, print_cost=True, print_learning_rate=False)

                # # base model+momentum optimization
                # parameters = L_layer_model_minibatch_momentum(X_train, y_train, layers_dims,
                #                                               learning_rate=best_params["learning_rate"],
                #                                               mini_batch_size=128, beta=0.99, num_epochs=8000,
                #                                               print_cost=True)

                # base model+momentum optimization+L2 regularization
                parameters = L_layer_model_minibatch_momentum_regularization(X_train, y_train, layers_dims,
                                                              learning_rate=learning_rate, plot_costs=False,
                                                              mini_batch_size=mini_batch_size, beta=beta,
                                                              num_epochs=1000, print_cost=True, _lambda=_lambda)

                pred_validation = predict(X_validation, y_validation, parameters)
                roc = roc_auc_score(y_validation[0], pred_validation[0].T)
                rocs.append(roc)
                terminal_print = '\tITERATION ' + str(i+1) + ' - FOLD ' + str(fold) + ' - ' + 'AUC_ROC : ' + str(roc)
                printm(terminal_print, color='yellow', switch=TERMCOLOR)

            roc_avg = np.average(rocs)
            printm('\nAverage AUC_ROC : ' + str(roc_avg), color='green', switch=TERMCOLOR)

            if roc_avg > best_auc_avg:
                best_auc_avg = roc_avg
                best_params["hl_0"] = hl_0
                best_params["hl_1"] = hl_1
                best_params["hl_2"] = hl_2
                best_params["learning_rate"] = learning_rate
                best_params["mini_batch_size"] = mini_batch_size
                best_params["beta"] = beta
                best_params["_lambda"] = _lambda

        print('\nBest ROC_AUC average after', N, 'iterations of KFold Cross Validations:', best_auc_avg)
        print('Best parameters: ', best_params, '\n')

    if not optimize_best_params:

        # use defined params
        printm('Using defined params on the whole training set ...', color='red', switch=TERMCOLOR)

        # Reshape the training and test examples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten / 255
        test_x = test_x_flatten / 255
        layers_dims = [np.shape(train_x)[0], best_params["hl_0"], best_params["hl_1"], best_params["hl_2"], 1]

        # # base model
        # parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=best_params["learning_rate"],
        #                                               num_iterations=2400, print_cost = True)

        # # base model+momentum optimization
        # parameters = L_layer_model_minibatch_momentum(train_x, train_y, layers_dims,
        #                                               learning_rate=best_params["learning_rate"],
        #                                               mini_batch_size=32, beta=0.9, num_epochs=2000, print_cost=True)
        #

        # base model+momentum optimzation+L2 regularization
        # parameters = L_layer_model_minibatch_momentum_regularization(train_x, train_y, layers_dims,
        #                                                              learning_rate=best_params["learning_rate"],
        #                                                              mini_batch_size=32,
        #                                                              beta=0.9, num_epochs=2000,
        #                                                              print_cost=True,
        #                                                              _lambda=0.1)

        # base model+momentum optimzation+L2 regularization
        parameters = L_layer_model_minibatch_momentum_regularization(train_x, train_y, layers_dims,
                                                      learning_rate=best_params["learning_rate"],
                                                      mini_batch_size=best_params["mini_batch_size"],
                                                      beta=best_params["beta"], num_epochs=2000, print_cost=True,
                                                      _lambda=best_params["_lambda"], plot_costs=True)


        pred_train = predict(train_x, train_y, parameters)
        pred_test = predict(test_x, test_y, parameters)
        roc = roc_auc_score(test_y[0], pred_test[0].T)
        printm('AUC_ROC : ' + str(roc), color='yellow', switch=TERMCOLOR)
        print_mislabeled_images(classes, test_x, test_y, pred_test)

    b = time.time()
    print(b - a, ' seconds')