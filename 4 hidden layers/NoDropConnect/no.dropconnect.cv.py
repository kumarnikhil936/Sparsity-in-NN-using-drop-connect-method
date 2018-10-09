#
# Author : Nikhil Kumar Jha
# 5 layer ( 4 hidden layers ) NN to classify MNIST data.
# No drop connect is being applied in this network.
#
# Last update made on: 8 June 2018
# Got a question ? Feel free to drop a mail at kumar.nikhil936@gmail.com
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Read the MNIST data set input
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.03
epochs = 150
batch_size = 100
accuracyTest = []
accuracyTrain = []

# declare training data placeholders (features/attributes)
# input x is 28 * 28 pixels
x = tf.placeholder(tf.float32, [None, 784])

# declare placeholders for output data (labels) - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# Declare the training and test data
train_x_all = mnist.train.images
train_y_all = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

# for L layers, there will be L-1 number of weights/bias tensors, and size depends on number of neurons present
# declare the weights connecting the input to hidden layer
w1 = tf.Variable(tf.random_normal([784, 400], stddev=0.03), name='w1')
# declare the variable for holding the bias terms
b1 = tf.Variable(tf.random_normal([400]), name='b1')

w2 = tf.Variable(tf.random_normal([400, 300], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([300]), name='b2')

w3 = tf.Variable(tf.random_normal([300, 200], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([200]), name='b3')

w4 = tf.Variable(tf.random_normal([200, 100], stddev=0.03), name='w4')
b4 = tf.Variable(tf.random_normal([100]), name='b4')

w5 = tf.Variable(tf.random_normal([100, 10], stddev=0.03), name='w5')
b5 = tf.Variable(tf.random_normal([10]), name='b5')

# calculate the output of hidden layer
# ReLU for hidden layers
hidden_out_1 = tf.add(tf.matmul(x, w1), b1)
hidden_out_1 = tf.nn.relu(hidden_out_1)

hidden_out_2 = tf.add(tf.matmul(hidden_out_1, w2), b2)
hidden_out_2 = tf.nn.relu(hidden_out_2)

hidden_out_3 = tf.add(tf.matmul(hidden_out_2, w3), b3)
hidden_out_3 = tf.nn.relu(hidden_out_3)

hidden_out_4 = tf.add(tf.matmul(hidden_out_3, w4), b4)
hidden_out_4 = tf.nn.relu(hidden_out_4)

y_ = tf.add(tf.matmul(hidden_out_4, w5), b5)
y_ = tf.nn.softmax(y_)


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
    np.savetxt("untrained_w1.csv", session.run(w1), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b1.csv", session.run(b1), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w2.csv", session.run(w2), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b2.csv", session.run(b2), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w3.csv", session.run(w3), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b3.csv", session.run(b3), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w4.csv", session.run(w4), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b4.csv", session.run(b4), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w5.csv", session.run(w5), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b5.csv", session.run(b5), delimiter="\t", fmt='%.4f')
    for epoch in range(epochs):
        total_batch = int(train_x_all.shape[0] / batch_size)
        avg_cost = 0
        for i in range(total_batch):
            #            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            batch_x = train_x_all[i * batch_size:(i + 1) * batch_size]
            batch_y = train_y_all[i * batch_size:(i + 1) * batch_size]
            _, c = session.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        accuracyTest.append(session.run(accuracy, feed_dict={x: test_x, y: test_y}))
        accuracyTrain.append(session.run(accuracy, feed_dict={x: train_x_all, y: train_y_all}))
    np.savetxt("trained_w1.csv", session.run(w1), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b1.csv", session.run(b1), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w2.csv", session.run(w2), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b2.csv", session.run(b2), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w3.csv", session.run(w3), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b3.csv", session.run(b3), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w4.csv", session.run(w4), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b4.csv", session.run(b4), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w5.csv", session.run(w5), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b5.csv", session.run(b5), delimiter="\t", fmt='%.4f')
    np.savetxt("accuracyTest.csv", accuracyTest, fmt='%.4f')
    np.savetxt("accuracyTrain.csv", accuracyTrain, fmt='%.4f')
