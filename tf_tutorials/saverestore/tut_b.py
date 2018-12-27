import tensorflow as tf

# create network
tf.reset_default_graph() # just to be safe
saver = tf.train.import_meta_graph('out/my_test_model.meta')

# load parameters
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('out/my_test_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('out/'))
    print(sess.run('w1:0')) # print saved value of w1
