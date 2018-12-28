import tensorflow as tf
a = tf.add(3, 5)
sess = tf.Session()
print sess.run(a)
sess.close()
