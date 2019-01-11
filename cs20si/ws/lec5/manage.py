import tensorflow as tf
import numpy as np
import os

# simple example for good management practice

# dummy dataset
def generate_dataset():
    # 3D feature, 1D label
    TRUE_W = np.array([2.0, -1.0, 3.0])
    TRUE_B = -3.0
    ds_size = 10000
    X = np.random.normal(0.0, 3.0, [ds_size, 3])
    noise = np.random.normal(0.0, 0.5, [ds_size])
    Y = np.einsum('nm,m->n', X, (TRUE_W)) + TRUE_B
    Y = Y + noise
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    print(ds.output_types)
    print(ds.output_shapes)
    return ds

def main():

    num_epochs = 50
    batch_size = 128
   
    ds = generate_dataset()

    # make batch iterator
    train_data = ds.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(
        train_data.output_types,
        train_data.output_shapes
    )
    X, Y = iterator.get_next()
    train_init = iterator.make_initializer(train_data)

    # define model, loss, optimizer
    with tf.variable_scope('linear') as scope:
        w = tf.get_variable('weight', [3], initializer=tf.random_normal_initializer())
        b = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
        Y_pred = tf.einsum('nm,m->n', X, w) + b
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss = tf.reduce_sum(tf.square(Y - Y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring saved model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("no checkpoint found")

        writer = tf.summary.FileWriter('graphs/manage', sess.graph)        
        for i in range(num_epochs):
            sess.run(train_init)
            total_loss = 0
            try:
                while True:
                    _, l = sess.run([optimizer, loss])
                    total_loss += l
            except tf.errors.OutOfRangeError:
                pass
            print("Training epoch {0} total loss {1}".format(i, total_loss))

            if(i + 1)%10 == 0:
                saver.save(sess, 'checkpoints/manage', i)

        w_learn, b_learn = sess.run([w, b])
        print("Learnt parameters: W:{0}, b:{1}".format(w_learn, b_learn))
        writer.close()

main()
