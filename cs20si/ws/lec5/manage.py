import tensorflow as tf
import numpy as np

# simple example for good management practice

# dummy dataset
def generate_dataset():
    # 3D feature, 1D label
    TRUE_W = [2.0, -1.0, 3.0]
    TRUE_B = -3.0
    ds_size = 10000
    X = np.random.normal(0.0, 3.0, [ds_size, 3])
    noise = np.random.normal(0.0, 0.5, [ds_size])
    Y = np.matmul(X, np.array(TRUE_W)) + noise
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    #print(ds.output_types)
    #print(ds.output_shapes)
    return ds

def main():
    ds = generate_dataset()
    iterator = ds.make_initializable_iterator()
    X, Y = iterator.get_next()
    
    # define model
    with tf.variable_scope('linear') as scope:
        w = tf.get_variable('weight', [3, 1], initializer=tf.random_normal_initializer())
        b = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
        Y_pred = tf.matmul(X, w) + b
    loss = tf.reduce_sum(tf.square(Y - Y_pred))

    optmizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    num_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            
            pass    

main()
