import h5py
import math
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
from skimage import transform, io

TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored
print('NumPy version', np.__version__)
print('Pandas version', pd.__version__)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 10000)


def load_dataset_hands(path):

    """
    Load hands dataset.

    :param data_path: <string> path to images directory
    :return all_images: <ndarray <float?>> image data
    """

    train_dataset = h5py.File(path + '/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path + '/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


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

    :return res_img: <ndarray <float>> image data resized
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


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    """
    Creates a list of random minibatches from (X, Y)

    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param mini_batch_size: size of the mini-batches, integer
    :param seed: this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    :return mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def one_hot_matrix(labels, C):

    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    :param labels: vector containing the labels
    :param C: number of classes, the depth of the one hot dimension

    :return one_hot: one hot matrix
    """

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, depth=C, axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    return one_hot


def create_placeholders(n_x, n_y):

    """
    Creates the placeholders for the tensorflow session. This will allow you to later pass your training data
    in when you run your session.

    :param n_x: scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    :param n_y: scalar, number of classes (from 0 to 5, so -> 6)

    :return X: placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    :return Y: placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    """

    X = tf.compat.v1.placeholder(tf.float32, name="X", shape=(n_x, None)) # you have to use None here because the
                                                                          # number of examples during train and test
                                                                          # is different
    Y = tf.compat.v1.placeholder(tf.float32, name="Y", shape=(n_y, None))

    return X, Y


def initialize_parameters(flat_depth_train, flat_depth_test):

    """
    Initializes parameters to build a neural network with tensorflow

    :param flat_depth_train: image height*width*depth
    :param flat_depth_test: number of unique test labels

    :return parameters: a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.random.set_seed(1)

    W1 = tf.compat.v1.get_variable("W1", [25, flat_depth_train], initializer=tf.keras.initializers.GlorotNormal(seed=1))
    b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.keras.initializers.GlorotNormal(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [flat_depth_test, 12], initializer=tf.keras.initializers.GlorotNormal(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [flat_depth_test, 1], initializer=tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):

    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    :param X: input dataset placeholder, of shape (input size, number of examples)
    :param parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                       the shapes are given in initialize_parameters

    :return Z3: the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.compat.v1.add(tf.compat.v1.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.compat.v1.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.compat.v1.add(tf.compat.v1.matmul(W2, A1), b2)  # Z2 = np.dot(W2, A1) + b2
    A2 = tf.compat.v1.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.compat.v1.add(tf.compat.v1.matmul(W3, A2), b3)  # Z3 = np.dot(W3, A2) + b3

    return Z3


def compute_cost(Z3, Y):

    """
    Computes the cost

    :param Z3: output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    :param Y: "true" labels vector placeholder, same shape as Z3

    :return cost: Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.compat.v1.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.compat.v1.transpose(Z3)
    labels = tf.compat.v1.transpose(Y)

    cost = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def convert_to_one_hot(Y, C):

    """
    Converts an array of (test) labels into a one-hot matrix of (test) labels

    :param Y: the array containing the labels
    :param C: the number of unique labels

    :return Y: one-hot conversion
    """

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y


def forward_propagation_for_predict(X, parameters):

    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    :param X: input dataset placeholder, of shape (input size, number of examples)
    :param parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                       the shapes are given in initialize_parameters

    :return Z3: the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.compat.v1.add(tf.compat.v1.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.compat.v1.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.compat.v1.add(tf.compat.v1.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.compat.v1.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.compat.v1.add(tf.compat.v1.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def tf_model(X_train, Y_train, X_test, Y_test, flat_depth_train, flat_depth_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):

    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    The general steps are as follows:
    1. Create Tensors (variables) that are not yet executed/evaluated.
    2. Write operations between those Tensors.
    3. Initialize your Tensors.
    4. Create a Session.
    5. Run the Session. This will run the operations you'd written above.

    :param X_train: training set, of shape (input size = 12288, number of training examples = 1080)
    :param Y_train: test set, of shape (output size = 6, number of training examples = 1080)
    :param X_test: training set, of shape (input size = 12288, number of training examples = 120)
    :param Y_test: test set, of shape (output size = 6, number of test examples = 120)
    :param learning_rate: learning rate of the optimization
    :param num_epochs: number of epochs of the optimization loop
    :param minibatch_size: size of a minibatch
    :param print_cost: True to print the cost every 100 epochs

    :return parameters: parameters learned by the model. They can then be used to predict.
    """

    tf.compat.v1.disable_eager_execution()

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.random.set_seed(1)
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(flat_depth_train, flat_depth_test)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1).minimize(cost)

    # Initialize all the variables
    init = tf.compat.v1.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(
                m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for i,minibatch in enumerate(minibatches):

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(Z3), tf.compat.v1.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


if __name__ == '__main__':

    a = time.time()
    print_packages_list()

    # load original dataset
    DATA_FOLDER = 'data/'
    write_h5_images(DATA_FOLDER)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset(DATA_FOLDER)
    flat_depth_X = np.prod(np.shape(train_x_orig)[-3:])
    flat_depth_Y = len(np.unique(test_y))

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

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    # Convert training and test labels to one hot matrices. This applies for multiclassification
    train_y = convert_to_one_hot(train_y.astype(int), 2)
    test_y = convert_to_one_hot(test_y.astype(int), 2)

    tf_model(train_x, train_y, test_x, test_y, flat_depth_X, flat_depth_Y, learning_rate=0.0001,
             num_epochs=2000, minibatch_size=32, print_cost=True)

    b = time.time()
    print('\n',b - a, ' seconds')


    # # Loading the dataset - hands
    # DATA_FOLDER = 'data/'
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset_hands(DATA_FOLDER)
    # flat_depth_X = np.prod(np.shape(X_train_orig)[-3:])
    #
    # # Flatten the training and test images
    # X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    # X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    #
    # # Normalize image vectors
    # X_train = X_train_flatten / 255.
    # X_test = X_test_flatten / 255.
    #
    # # Convert training and test labels to one hot matrices
    # Y_train = convert_to_one_hot(Y_train_orig, 6)
    # Y_test = convert_to_one_hot(Y_test_orig, 6)
    # flat_depth_Y = len(np.unique(Y_test_orig))
    #
    #
    # print("number of training examples = " + str(X_train.shape[1]))
    # print("number of test examples = " + str(X_test.shape[1]))
    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))
    #
    # tf_model(X_train, Y_train, X_test, Y_test, flat_depth_X, flat_depth_Y, learning_rate=0.0001,
    #       num_epochs=100, minibatch_size=32, print_cost=True)
    #
    # b = time.time()
    # print(b - a, ' seconds')