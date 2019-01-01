# placeholders
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
    print(sess.run(c, feed_dict={a: [1, 2, 3], b: [2, 2, 2]})) # dummy value!
