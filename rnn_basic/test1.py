import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell


def main1():

    cell = BasicRNNCell(2)

    X = tf.constant([[1.0], [2.0]])
    
    init_state = tf.constant([[3.0] , [4.0]])
    out, state = cell(X, init_state)

    writer = tf.summary.FileWriter('./debug_out')
    writer.add_graph(tf.get_default_graph())
    writer.flush()
    pass

main1()
