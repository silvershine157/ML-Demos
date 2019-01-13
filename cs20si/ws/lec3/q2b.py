import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

learning_rate = 0.01
batch_size = 128
n_epochs = 50
n_test = 10000

def fc_layer(x, output_dim, scope_str, activation=True):
    with tf.variable_scope(scope_str) as scope:
        w = tf.get_variable('weight', (x.shape[1], output_dim),
            initializer=tf.random_normal_initializer(0.0, 0.01))
        b = tf.get_variable('bias', (1, output_dim),
            initializer=tf.random_normal_initializer(0.0, 0.01))
        y = tf.matmul(x, w) + b
        if(not activation):
            return y
        a = tf.nn.relu(tf.matmul(x, w) + b)
    return a

def main():
    # read data
    train, test = get_dataset()
    train_batch = train.batch(batch_size)
    test_batch = test.batch(batch_size)

    # one iterator, init with different datasets
    iterator = tf.data.Iterator.from_structure(
        train_batch.output_types,
        train_batch.output_shapes
    )
    img, label = iterator.get_next()
    train_init = iterator.make_initializer(train_batch)
    test_init = iterator.make_initializer(test_batch)

    # create model
    h1 = fc_layer(img, 1000, 'fc1')
    h2 = fc_layer(h1, 500, 'fc2')
    logits = fc_layer(h2, 10, 'fc3', activation=False)
    
    '''
    # w: feature_size x n_classes matrix
    # b: 1 x n_classes vector

    w = tf.get_variable('weight', (784, 10),
        initializer=tf.random_normal_initializer(0.0, 0.01))
    b = tf.get_variable('bias', (1, 10),
        initializer=tf.random_normal_initializer(0.0, 0.01))

    logits = tf.matmul(img, w) + b
    '''

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=label
    )
    loss = tf.reduce_sum(cross_ent, name='loss')
    
    # create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # calculate accuracy
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    #writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())

        # train model
        for i in range(n_epochs):
            sess.run(train_init) # draw training batch
            total_loss = 0
            n_batches = 0
            try:
                while True:
                    _, l = sess.run([optimizer, loss])
                    total_loss += l
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        print('Total time: {0} seconds'.format(time.time() - start_time))

        # test model
        sess.run(test_init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(accuracy)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy {0}'.format(total_correct_preds/n_test))
    # writer.close()


def get_dataset():
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_label))
    return train_ds, test_ds

main()
