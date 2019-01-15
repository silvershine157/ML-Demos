import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_test = 10000

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters],
            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters],
            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1], padding=padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out


def get_dataset():
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_label))
    return train_ds, test_ds


class ConvNet(object):

    def __init__(self):
        self.keep_prob = tf.constant(0.75)
        self.training = True
        self.n_classes = 10

    def get_data(self):
        train, test = get_dataset()
        train_batch = train.batch(batch_size)
        test_batch = test.batch(batch_size)
        iterator = tf.data.Iterator.from_structure(
            train_batch.output_types,
            train_batch.output_shapes
        )
        self.img, self.label = iterator.get_next()
        self.train_init = iterator.make_initializer(train_batch)
        self.test_init = iterator.make_initializer(test_batch)
        self.img = tf.reshape(self.img, shape=[-1, 28, 28, 1])

    def create_model(self):
        conv1 = conv_relu(
            inputs=self.img,
            filters=32,
            k_size=5,
            stride=1,
            padding='SAME',
            scope_name='conv1'
        )
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(
            inputs=pool1,
            filters=64,
            k_size=5,
            stride=1,
            padding='SAME',
            scope_name='conv2'
        )
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.nn.relu(fully_connected(pool2, 1024, 'fc'))
        #dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training,
        #    name='dropout')
        self.logits = fully_connected(fc, self.n_classes, 'logits')


    def create_loss(self):
        with tf.name_scope('loss'):
            cross_ent = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label
            )
            self.loss = tf.reduce_sum(cross_ent)

    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def create_accuracy(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


def debug():
    model = ConvNet()
    model.get_data()
    model.create_model()
    model.create_loss()
    print(model.logits.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.train_init)

        logits = sess.run(model.logits)
        print(logits.shape)

def main():

    # initialize model
    model = ConvNet()
    model.get_data()
    model.create_model()
    model.create_loss()
    model.create_optimizer()
    model.create_accuracy()

    writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())
    with tf.Session() as sess:
        # train
        model.training = True
        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            sess.run(model.train_init)
            total_loss = 0
            n_batches = 0
            try:
                while True:
                    _, l = sess.run([model.optimizer, model.loss])
                    total_loss += l
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
            
        # test
        model.training = False
        sess.run(model.test_init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(model.accuracy)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy {0}'.format(float(total_correct_preds)/n_test))
    writer.close()


main()
