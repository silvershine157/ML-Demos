import tensorflow as tf

### TF Low Level API Tutorial: Variables

# tf.Variable value change by ops, outside context of single session.run

def main1():
    # make variable with name and shape
    my_var = tf.get_variable("my_var", [1, 2, 3])

    # initialize with zeros
    zero_var = tf.get_variable("zero_var", [3,3], initializer=tf.zeros_initializer)

    # initialize with tensor value (no shape given)
    other_var = tf.get_variable("other_var", dtype=tf.int32, initializer=tf.constant([23, 42]))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run({"my_var": my_var, "zero_var": zero_var, "other_var": other_var}))

def main2():
    pass

main1()
