import tensorflow as tf

def main():
    v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        inc_v1.op.run()
        dec_v2.op.run()
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: "+save_path)



main()
