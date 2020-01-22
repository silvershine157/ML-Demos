import tensorflow as tf

x = tf.Variable(2.0)
y = 2.0 * (x ** 3)

grad_y = tf.gradients(y, x)
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_y))
