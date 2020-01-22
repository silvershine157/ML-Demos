import tensorflow as tf
a = tf.linspace(2.0, 3.0, 10, name="avec")
print(a)
with tf.Session() as sess:
    res = sess.run(a)
    print(res)
