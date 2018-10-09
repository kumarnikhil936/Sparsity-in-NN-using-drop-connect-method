#
# Author : Nikhil Kumar Jha
# 10 layer ( 9 hidden layers ) NN to classify MNIST data.
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
w1 = tf.Variable(tf.random_normal([784, 200], stddev=0.03), name='w1')
# declare the variable for holding the bias terms
b1 = tf.Variable(tf.random_normal([200]), name='b1')

w2 = tf.Variable(tf.random_normal([200, 150], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([150]), name='b2')

w3 = tf.Variable(tf.random_normal([150, 140], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([140]), name='b3')

w4 = tf.Variable(tf.random_normal([140, 130], stddev=0.03), name='w4')
b4 = tf.Variable(tf.random_normal([130]), name='b4')

w5 = tf.Variable(tf.random_normal([130, 120], stddev=0.03), name='w5')
b5 = tf.Variable(tf.random_normal([120]), name='b5')

w6 = tf.Variable(tf.random_normal([120, 80], stddev=0.03), name='w6')
b6 = tf.Variable(tf.random_normal([80]), name='b6')

w7 = tf.Variable(tf.random_normal([80, 70], stddev=0.03), name='w7')
b7 = tf.Variable(tf.random_normal([70]), name='b7')

w8 = tf.Variable(tf.random_normal([70, 60], stddev=0.03), name='w8')
b8 = tf.Variable(tf.random_normal([60]), name='b8')

w9 = tf.Variable(tf.random_normal([60, 50], stddev=0.03), name='w9')
b9 = tf.Variable(tf.random_normal([50]), name='b9')

w10 = tf.Variable(tf.random_normal([50, 10], stddev=0.03), name='w10')
b10 = tf.Variable(tf.random_normal([10]), name='b10')

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

hidden_out_5 = tf.add(tf.matmul(hidden_out_4, w5), b5)
hidden_out_5 = tf.nn.relu(hidden_out_5)

hidden_out_6 = tf.add(tf.matmul(hidden_out_5, w6), b6)
hidden_out_6 = tf.nn.relu(hidden_out_6)

hidden_out_7 = tf.add(tf.matmul(hidden_out_6, w7), b7)
hidden_out_7 = tf.nn.relu(hidden_out_7)

hidden_out_8 = tf.add(tf.matmul(hidden_out_7, w8), b8)
hidden_out_8 = tf.nn.relu(hidden_out_8)

hidden_out_9 = tf.add(tf.matmul(hidden_out_8, w9), b9)
hidden_out_9 = tf.nn.relu(hidden_out_9)

y_ = tf.add(tf.matmul(hidden_out_9, w10), b10)
y_ = tf.nn.softmax(y_)  # Softmax activation for output layer

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
    np.savetxt("untrained_w6.csv", session.run(w6), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b6.csv", session.run(b6), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w7.csv", session.run(w7), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b7.csv", session.run(b7), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w8.csv", session.run(w8), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b8.csv", session.run(b8), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w9.csv", session.run(w9), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b9.csv", session.run(b9), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_w10.csv", session.run(w10), delimiter="\t", fmt='%.4f')
    np.savetxt("untrained_b10.csv", session.run(b10), delimiter="\t", fmt='%.4f')
    for epoch in range(epochs):
        total_batch = int(train_x_all.shape[0] / batch_size)
        avg_cost = 0
        for i in range(total_batch):
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
    np.savetxt("trained_w6.csv", session.run(w6), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b6.csv", session.run(b6), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w7.csv", session.run(w7), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b7.csv", session.run(b7), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w8.csv", session.run(w8), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b8.csv", session.run(b8), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w9.csv", session.run(w9), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b9.csv", session.run(b9), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_w10.csv", session.run(w10), delimiter="\t", fmt='%.4f')
    np.savetxt("trained_b10.csv", session.run(b10), delimiter="\t", fmt='%.4f')
    np.savetxt("accuracyTest.csv", accuracyTest, fmt='%.4f')
    np.savetxt("accuracyTrain.csv", accuracyTrain, fmt='%.4f')
