#
# Author : Nikhil Kumar Jha
# 4 layer ( 3 hidden layers ) NN to classify MNIST data
# Drop connect regularization is implemented here using the dropConnect function.
# Here applying the dropConnect between hidden layer 2 and hidden layer 3 .
# Using Cross validation technique to get reliable results.
#
# Last update made on: 25 May 2018
# Got a question ? Feel free to drop a mail at kumar.nikhil936@gmail.com
#

import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Read the MNIST data set input
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.03
epochs = 20
batch_size = 500
accuracyTrain = []
accuracyTest = []
cvResult = []


# dropConnect function
# Arguments: w - Weight vector
#            p - Percentage value (0 - 0.99) stating how many weights to preserve (1 - drop percentage)
def dropConnect(w, p):
    wShape = w.shape
    n = np.prod(wShape)
    maskBits = np.random.choice(np.arange(n), int(np.floor((1-p)*n)), replace=False)
    w = w.flatten()
    w[maskBits] = 0
    w = w.reshape(wShape)
    return w


# run_train function
# Arguments: session - tensorflow session variable
#            train_x - feature space / attributes
#            train_y - labels
def run_train(session, train_x, train_y):
    print("** Start training **")
    session.run(init_op)
    for epoch in range(epochs):
        total_batch = int(train_x.shape[0] / batch_size)
        for i in range(total_batch):
            batch_x = train_x[i * batch_size:(i + 1) * batch_size]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size]
            _, c = session.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        print("Epoch #%d cost=%f" % (epoch, c))


# cross_validate function
# Arguments: session - tensorflow session variable
#            splitsize - number of folds to be used for validation
def cross_validate(session, split_size=10):
    results = []
    kf = KFold(n_splits=split_size)
    for train_idx, val_idx in kf.split(train_x_all, train_y_all):
        train_x = train_x_all[train_idx]
        train_y = train_y_all[train_idx]
        val_x = train_x_all[val_idx]
        val_y = train_y_all[val_idx]
        # run_train(session, train_x, train_y)
        results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
    return results


# declare training data placeholders (features/attributes)
# input x is 28 * 28 pixels
x = tf.placeholder(tf.float32, [None, 784])

# declare placeholders for output data (labels) - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

for keep_percent in np.arange(1, 0.0, -0.05):  # dropping keep_percent in steps of 0.05 in every iteration
    # Declare the training and test data
    train_x_all = mnist.train.images
    train_y_all = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels

    # for L layers, there will be L-1 number of weights/bias tensors, and size depends on number of neurons present

    # declare the weights connecting the input to hidden layer
    # declare the variable for holding the bias terms
    w1 = tf.Variable(np.genfromtxt('trained_w1.csv', delimiter="\t"), dtype=tf.float32, name='w1')
    b1 = tf.Variable(np.genfromtxt('trained_b1.csv', delimiter="\t"), dtype=tf.float32, name='b1')
    w2 = tf.Variable(np.genfromtxt('trained_w2.csv', delimiter="\t"), dtype=tf.float32, name='w2')
    b2 = tf.Variable(np.genfromtxt('trained_b2.csv', delimiter="\t"), dtype=tf.float32, name='b2')
    w3 = tf.Variable(dropConnect(np.genfromtxt('trained_w3.csv', delimiter="\t"), p=keep_percent), dtype=tf.float32, name='w3')
    b3 = tf.Variable(np.genfromtxt('trained_b3.csv', delimiter="\t"), dtype=tf.float32, name='b3')
    w4 = tf.Variable(np.genfromtxt('trained_w4.csv', delimiter="\t"), dtype=tf.float32, name='w4')
    b4 = tf.Variable(np.genfromtxt('trained_b4.csv', delimiter="\t"), dtype=tf.float32, name='b4')

    # calculate the output of hidden layer and the output layer
    hidden_out_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

    hidden_out_2 = tf.nn.relu(tf.add(tf.matmul(hidden_out_1, w2), b2))

    hidden_out_3 = tf.nn.relu(tf.add(tf.matmul(hidden_out_2, w3), b3))

    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out_3, w4), b4))

    # convert the output y_ to a clipped version, limited between 1e-10 to 0.999999.
    # This is to make sure that we never get a case were we have a log(0) operation occurring during training
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

    # Calculating cross_entropy
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # Using the gradient descent optimiser to minimise the cross entropy
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # setup the initialization operator
    init_op = tf.global_variables_initializer()

    # define accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
        session.run(init_op)
        print("Nodes keep percentage is : ", keep_percent)
        cvResult.append(cross_validate(session))
        accuracyTest.append(session.run(accuracy, feed_dict={x: test_x, y: test_y}))
        accuracyTrain.append(session.run(accuracy, feed_dict={x: train_x_all, y: train_y_all}))


np.savetxt("dl2AccuracyTestNoRetrain.csv", accuracyTest, delimiter="\n", fmt='%.4f')
np.savetxt("dl2AccuracyTrainNoRetrain.csv", accuracyTrain, delimiter="\n", fmt='%.4f')
np.savetxt("dl2cvResultsNoRetrain.csv", cvResult, delimiter="\t", fmt='%.4f')

# np.savetxt("dl2AccuracyTestWithRetrain.csv", accuracyTest, delimiter="\n", fmt='%.4f')
# np.savetxt("dl2AccuracyTrainWithRetrain.csv", accuracyTrain, delimiter="\n", fmt='%.4f')
# np.savetxt("dl2cvResultsWithRetrain.csv", cvResult, delimiter="\t", fmt='%.4f')
