# Author : Nikhil Kumar Jha
# 3 layer NN to classify MNIST data

# Using drop connect regularization

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.1
epochs = 10
batch_size = 100

# declare training data placeholders 
# input x is 28 * 28 pixels 
x = tf.placeholder(tf.float32, [None, 784])

# declare placeholders for output data - 10 digits 
y = tf.placeholder(tf.float32, [None, 10])

keep_probab = 0.01

# for L layers, there will be L-1 number of weights/bias tensors
# declare the weights connecting the input to hidden layer
w1 = tf.Variable(tf.random_normal([784, 600], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([600]), name='b1')
w2 = tf.Variable(tf.random_normal([600, 400], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([400]), name='b2')
w3 = tf.Variable(tf.random_normal([400, 10], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([10]), name='b3')

'''
def dropConnect(w, p):
    m_vector = tf.multinomial(tf.log([[1-p, p]]), np.prod(w.shape))
    m = tf.reshape(m_vector, w.shape)
    m = tf.cast(m, tf.float32)
    return m * w
'''

'''
def dropConnect(w, p):
    return tf.nn.dropout(w, keep_prob=p) * p
'''


# calculate the output of hidden layer
# ReLU for hidden layers
hidden_out_1 = tf.add(tf.matmul(x, w1), b1)
hidden_out_1 = tf.nn.relu(hidden_out_1)
hidden_out_1 = tf.nn.dropout(hidden_out_1, keep_prob=keep_probab)

hidden_out_2 = tf.add(tf.matmul(hidden_out_1, w2), b2)
hidden_out_2 = tf.nn.relu(hidden_out_2)
hidden_out_2 = tf.nn.dropout(hidden_out_2, keep_prob=keep_probab)

y_ = tf.add(tf.matmul(hidden_out_2, w3), b3)
# Softmax activation for output layer
y_ = tf.nn.softmax(y_)

# convert the output y_ to a clipped version, limited between 1e-10 to 0.999999.
# This is to make sure that we never get a case were we have a log(0) operation occurring during training
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# setup the initialization operator
init_op = tf.global_variables_initializer()

# define accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the training session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            c, _ = sess.run([cross_entropy, optimiser], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
    print("Accuracy =", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # print(sess.run(w1))


