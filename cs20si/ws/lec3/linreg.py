import tensorflow as tf
import numpy as np

DATA_FILE = "../data/birth_life_2010.txt"

def read_data(fname):
    f = open(fname, 'r')
    L = f.readlines()
    f.close()
    L.pop(0)
    n_samples = len(L)
    data = np.zeros((n_samples, 2))
    for i in range(n_samples):
        row = L[i].strip().split()
        data[i, 0] = float(row[-2])
        data[i, 1] = float(row[-1])
    return data, n_samples

def main():
    data, n_samples = read_data(DATA_FILE)

    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

    w = tf.get_variable('weights', initializer=tf.constant(0.0))
    b = tf.get_variable('bias', initializer=tf.constant(0.0))

    Y_predicted = w * X + b

    loss = tf.square(Y - Y_predicted, name='loss')
    
    # gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train model
        for i in range(100):
            for x, y in data:
                sess.run(optimizer, feed_dict={X:x, Y:y})
        w_out, b_out = sess.run([w, b])
    print(w_out, b_out)

def analytic():
    # this time, analytically solve linear regression
    data, n_samples = read_data(DATA_FILE)
    
    X = np.hstack([data[:, 0].reshape((n_samples, 1)), np.ones((n_samples, 1))])
    Y = data[:, 1].reshape((n_samples, 1))

    Xt = np.transpose(X)
    invXtX = np.linalg.inv(np.matmul(Xt, X))
    W = np.matmul(np.matmul(invXtX, Xt), Y)
    print(W)

analytic()
