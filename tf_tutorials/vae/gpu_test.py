import tensorflow as tf

def gpu_wrapper():
    if(tf.test.is_gpu_available()):
        with(tf.device('GPU:3')):
            print("Running with GPU:3")
    else:
        print("No GPU available!")

gpu_wrapper()
