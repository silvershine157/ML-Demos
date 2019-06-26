import tensorflow as tf
W = tf.get_variable("W", initializer=tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
