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

class LinRegModel(object):
    def __init__(self):
        self.batch_size = 128
        self.test_size = 1000
        pass

    def setup_data(self):
        ds = generate_dataset()
        test_ds = ds.take(self.test_size)
        train_ds = ds.skip(self.test_size)
        test_data = test_ds.batch(self.batch_size)
        train_data = train_ds.batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(
            train_data.output_types,
            train_data.output_shapes
        )
        self.X, self.Y = iterator.get_next()
        self.test_init = iterator.make_initializer(test_data)
        self.train_init = iterator.make_initializer(train_data)

    def create_loss(self):
        # define model, loss, optimizer
        with tf.variable_scope('linear') as scope:
            w = tf.get_variable('weight', [3], initializer=tf.random_normal_initializer())
            b = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
            Y_pred = tf.einsum('nm,m->n', self.X, w) + b
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.loss = tf.reduce_sum(tf.square(self.Y - Y_pred))
        
    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        
    def create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()


def main():

    num_epochs = 50

    # setup model
    model = LinRegModel()
    model.setup_data()
    model.create_loss()
    model.create_optimizer()
    model.create_summaries()
    
    saver = tf.train.Saver()
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
            sess.run(model.train_init)
            total_loss = 0
            try:
                while True:
                    _, l, summary = sess.run([model.optimizer, model.loss, model.summary_op])
                    writer.add_summary(summary, global_step=i) # this is bad
                    total_loss += l
            except tf.errors.OutOfRangeError:
                pass
            print("Training epoch {0} total loss {1}".format(i, total_loss))

            if(i + 1)%10 == 0:
                saver.save(sess, 'checkpoints/manage', i)

        writer.close()


main()



