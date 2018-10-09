#
# Author : Nikhil Kumar Jha
# 8 layer ( 7 hidden layers ) FFNN to classify MNIST data
# Drop connect regularization is implemented here using the dropConnect function.
# Here applying the dropConnect to all the layers except the input layer.
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
epochs = 50
batch_size = 500
accuracyTrain = []
accuracyTest = []
cvResult = []


# dropConnect function
# Arguments: w1, w2, w3, w4, w5, w6, w7, w8 - Weight vectors
#            p - Percentage value (0.05 - 1) stating how much percentage of weights to preserve (1 - drop percentage)
def dropConnect(w1, w2, w3, w4, w5, w6, w7, w8, p):
    w1s = w1.shape
    w2s = w2.shape
    w3s = w3.shape
    w4s = w4.shape
    w5s = w5.shape
    w6s = w6.shape
    w7s = w7.shape
    w8s = w8.shape
    n = np.prod(w1s) + np.prod(w2s) + np.prod(w3s) + np.prod(w4s) + np.prod(w5s) + np.prod(w6s) + np.prod(w7s) + np.prod(w8s)
    maskBits = np.random.choice(np.arange(n), int(np.floor((1-p)*n)), replace=False)
    wf = np.concatenate((w1.flatten(), w2.flatten(), w3.flatten(), w4.flatten(), w5.flatten(), w6.flatten(), w7.flatten(), w8.flatten()), axis=0)
    wf[maskBits] = 0
    w1m = wf[0:np.prod(w1s)].reshape(w1s)
    w2m = wf[np.prod(w1s):(np.prod(w1s)+np.prod(w2s))].reshape(w2s)
    w3m = wf[(np.prod(w1s)+np.prod(w2s)):(np.prod(w1s)+np.prod(w2s)+np.prod(w3s))].reshape(w3s)
    w4m = wf[(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)):(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s))].reshape(w4s)
    w5m = wf[(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)):(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s))].reshape(w5s)
    w6m = wf[(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s)):(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s)+np.prod(w6s))].reshape(w6s)
    w7m = wf[(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s)+np.prod(w6s)):(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s)+np.prod(w6s)+np.prod(w7s))].reshape(w7s)
    w8m = wf[(np.prod(w1s)+np.prod(w2s)+np.prod(w3s)+np.prod(w4s)+np.prod(w5s)+np.prod(w6s)+np.prod(w7s)):].reshape(w8s)
    return w1m, w2m, w3m, w4m, w5m, w6m, w7m, w8m


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
#            split_size - number of folds to be used for validation
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

    # declare the variable for weights connecting the input to hidden layer
    # declare the variable for holding the bias terms
    w1 = np.genfromtxt('trained_w1.csv', delimiter="\t")
    b1 = np.genfromtxt('trained_b1.csv', delimiter="\t")
    w2 = np.genfromtxt('trained_w2.csv', delimiter="\t")
    b2 = np.genfromtxt('trained_b2.csv', delimiter="\t")
    w3 = np.genfromtxt('trained_w3.csv', delimiter="\t")
    b3 = np.genfromtxt('trained_b3.csv', delimiter="\t")
    w4 = np.genfromtxt('trained_w4.csv', delimiter="\t")
    b4 = np.genfromtxt('trained_b4.csv', delimiter="\t")
    w5 = np.genfromtxt('trained_w5.csv', delimiter="\t")
    b5 = np.genfromtxt('trained_b5.csv', delimiter="\t")
    w6 = np.genfromtxt('trained_w6.csv', delimiter="\t")
    b6 = np.genfromtxt('trained_b6.csv', delimiter="\t")
    w7 = np.genfromtxt('trained_w7.csv', delimiter="\t")
    b7 = np.genfromtxt('trained_b7.csv', delimiter="\t")
    w8 = np.genfromtxt('trained_w8.csv', delimiter="\t")
    b8 = np.genfromtxt('trained_b8.csv', delimiter="\t")

    w1, w2, w3, w4, w5, w6, w7, w8 = dropConnect(w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6, w7=w7, w8=w8, p=keep_percent)

    w1 = tf.Variable(w1, dtype=tf.float32, name='w1')
    w2 = tf.Variable(w2, dtype=tf.float32, name='w2')
    w3 = tf.Variable(w3, dtype=tf.float32, name='w3')
    w4 = tf.Variable(w4, dtype=tf.float32, name='w4')
    w5 = tf.Variable(w5, dtype=tf.float32, name='w5')
    w6 = tf.Variable(w6, dtype=tf.float32, name='w6')
    w7 = tf.Variable(w7, dtype=tf.float32, name='w7')
    w8 = tf.Variable(w8, dtype=tf.float32, name='w8')

    # calculate the output of hidden layer and the output layer
    hidden_out_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

    hidden_out_2 = tf.nn.relu(tf.add(tf.matmul(hidden_out_1, w2), b2))

    hidden_out_3 = tf.nn.relu(tf.add(tf.matmul(hidden_out_2, w3), b3))

    hidden_out_4 = tf.nn.relu(tf.add(tf.matmul(hidden_out_3, w4), b4))

    hidden_out_5 = tf.nn.relu(tf.add(tf.matmul(hidden_out_4, w5), b5))

    hidden_out_6 = tf.nn.relu(tf.add(tf.matmul(hidden_out_5, w6), b6))

    hidden_out_7 = tf.nn.relu(tf.add(tf.matmul(hidden_out_6, w7), b7))

    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out_7, w8), b8))

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

np.savetxt("uniformDropconnectAccuracyTestNoRetrain.csv", accuracyTest, delimiter="\n", fmt='%.4f')
np.savetxt("uniformDropconnectAccuracyTrainNoRetrain.csv", accuracyTrain, delimiter="\n", fmt='%.4f')
np.savetxt("uniformDropconnectcvResultsNoRetrain.csv", cvResult, delimiter="  ", fmt='%.4f')

# np.savetxt("uniformDropconnectAccuracyTestWithRetrain.csv", accuracyTest, delimiter="\n", fmt='%.4f')
# np.savetxt("uniformDropconnectAccuracyTrainWithRetrain.csv", accuracyTrain, delimiter="\n", fmt='%.4f')
# np.savetxt("uniformDropconnectcvResultsWithRetrain.csv", cvResult, delimiter="  ", fmt='%.4f')
