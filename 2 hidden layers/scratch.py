import tensorflow as tf
import numpy as np


def dropConnect(w, p):
    wShape = w.shape
    print("\nShape of w is ", wShape)
    print("\nw\n", w, "\n")
    n = np.prod(wShape)
    maskBits = np.random.choice(np.arange(n), int(np.floor((1-p)*n)), replace=False)
    print("Total number of elements in the matrix is ", n, " and for keep percentage ", p, ", these bits will be masked ", maskBits)
    wf = w.flatten()
    print("\nwf\n ", wf)
    wf[maskBits] = 0
    print("\nAfter masking \n", wf)
    wm = wf.reshape(wShape)
    print("\njunk\n", wm)
    # np.savetxt("mask.csv", maskBits, delimiter="\t", fmt='%.2f')
    return wm


def dropConnect1(w1, w2, w3, p):
    w1s = w1.shape
    w2s = w2.shape
    w3s = w3.shape
    print("\nShape of w1 is ", w1s)
    print("\nw1", w1, "\n")
    print("Shape of w2 is ", w2s)
    print("\nw2", w2, "\n")
    print("Shape of w3 is ", w3s)
    print("\nw3", w3, "\n")
    n = np.prod(w1s) + np.prod(w2s) + np.prod(w3s)
    maskBits = np.random.choice(np.arange(n), int(np.floor((1-p)*n)), replace=False)
    print("Total number of elements in the matrix is ", n, " and for keep percentage ", p, ", these bits will be masked ", maskBits)
    wf = np.concatenate((w1.flatten(), w2.flatten(), w3.flatten()), axis=0)
    print("\nwf\n ", wf)
    wf[maskBits] = 0
    print("\nAfter masking \n", wf)
    w1m = wf[0:np.prod(w1s)].reshape(w1s)
    w2m = wf[np.prod(w1s):(np.prod(w1s)+np.prod(w2s))].reshape(w2s)
    w3m = wf[(np.prod(w1s)+np.prod(w2s)):].reshape(w3s)
    # np.savetxt("mask.csv", maskBits, delimiter="\t", fmt='%.2f')
    return w1m, w2m, w3m


p = 0.2

# w = tf.Variable(tf.random_normal([7, 6], stddev=0.03), name='w')
# w = np.random.normal(size=(6, 6))
# w = dropConnect(w, p=p)
# print("After dropConnect, w matrix is ", w)

# junk = tf.Variable(dropConnect(np.genfromtxt('trained_w3.csv', delimiter="\t"), p=0.95), dtype=tf.float32)

w = np.random.random((5, 2))
junk = tf.Variable(dropConnect(w, p), dtype=tf.float32)

w1 = np.random.random((5, 2))
w2 = np.random.random((4, 2))
w3 = np.random.random((2, 3))
junk1, junk2, junk3 = dropConnect1(w1, w2, w3, p)
print("\njunk1\n", junk1, "\n")
print("\njunk2\n", junk2, "\n")
print("\njunk3\n", junk3, "\n")
junk1 = tf.Variable(junk1, dtype=tf.float32)
junk2 = tf.Variable(junk2, dtype=tf.float32)
junk3 = tf.Variable(junk3, dtype=tf.float32)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    # np.savetxt("junk.csv", session.run(junk), delimiter="\t", fmt='%.4f')
    # np.savetxt("junk1.csv", session.run(junk1), delimiter="\t", fmt='%.4f')
    # np.savetxt("junk2.csv", session.run(junk2), delimiter="\t", fmt='%.4f')

