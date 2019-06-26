import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print sess.run(x)
writer.close()
