from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

### Compuatational Graph, Session, TensorBoard

# <Computational Graph> (tf.Graph)
# tf.Operation : correspond to nodes
# tf.Tensor : correspond to edges

def main1():
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)
    total = a + b
    print(a)
    print(b)
    print(total)
    # each operation is given a unique name

    # save graph to tensorboard summary file
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    # Use
    # $ tensorboard --logdir .
    # to run tensorboard
    
    # tf.Session: runs tf.Graph
    sess = tf.Session()
    print(sess.run(total)) # evaluate tensor
    print(sess.run({'ab':(a, b), 'total':total})) # or multiple tensors (can be any tuple/dict combination)


### Feeding Data

def main2():
    # placeholder: feeding simple data
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    sess = tf.Session()
    print(sess.run(z, feed_dict = {x: 3, y: 4.5}))
    print(sess.run(z, feed_dict = {x: [1, 3], y: [2, 4.5]}))

def main3():
    # tf.data is more preferred
    my_data = [
        [0, 1,],
        [2, 3,],
        [4, 5,],
        [6, 7,],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()

    # print each row
    sess = tf.Session()
    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break

def main4():
    # initialize iterator for stateful operations
    r = tf.random_normal([10, 3])
    dataset = tf.data.Dataset.from_tensor_slices(r)
    iterator = dataset.make_initializable_iterator()
    next_row = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_row))
        except tf.errors.OutOfRangeError:
            break

def main5():
    sess = tf.Session()

    # create layers
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    # intialize all variable
    init = tf.global_variables_initializer()
    sess.run(init)

    # execute layers
    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

# let me skip the feature column example

def main6():
    # define data
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    
    # define model
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print("Prediction before training:")
    print(sess.run(y_pred))

    # define loss
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print("Loss before training:")
    print(sess.run(loss))

    # training
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)
    # 'train' is an op, not a tensor (no value produced)
    # so evaluate 'loss' each time
    print("Start training")
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)
    
    print("Prediction after training")
    print(sess.run(y_pred))
    
    
main6()
