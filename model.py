import tensorflow as tf
# due to resource constraints, the model input will be shrunk dramatically from input image
# leading our input image to be < 100x100 pixels, specifically the shape of 68x68x3 for this model

# the "dl" stands for "doomlasagna"
def simple_dl_classifier(input_placeholder):
    conv1 = tf.layers.conv2d(
        inputs=input_placeholder,
        kernel_size=[5, 5],
        strides=[1, 1],
        filters=8,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        padding='VALID'
    )  # [1, 68, 68, 3] -> [1, 64, 64, 8]
    mp_1 = tf.nn.max_pool2d(
        input=conv1,
        ksize=[2,2],
        strides=[2,2],
        padding='VALID'
    )  # [1, 64, 64, 8] -> [1, 32, 32, 8]
    conv2 = tf.layers.conv2d(
        inputs=mp_1,
        kernel_size=[5, 5],
        strides=[1, 1],
        filters=24,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        padding='VALID'
    )  # [1, 32, 32, 8] -> [1, 28, 28, 24]
    mp_2 = tf.nn.max_pool2d(
        input=conv2,
        ksize=[2, 2],
        strides=[2, 2],
        padding='VALID'
    )  # [1, 28, 28, 24] -> [1, 14, 14, 24]
    conv3 = tf.layers.conv2d(
        inputs=mp_2,
        kernel_size=[5, 5],
        strides=[1, 1],
        filters=64,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        padding='VALID'
    )  # [1, 14, 14, 24] -> [1, 10, 10, 64]
    mp_3 = tf.nn.max_pool2d(
        input=conv3,
        ksize=[2, 2],
        strides=[2, 2],
        padding='VALID'
    )  # [1, 10, 10, 64] -> [1, 5, 5, 64]
    fully_connected_1 = tf.layers.conv2d(  # really, technically equivalent to a squeeze followed by a FC layer
        inputs=mp_3,
        kernel_size=[5, 5],
        strides=[1, 1],
        filters=128,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        padding='VALID'
    )  # [1, 5, 5, 64] -> [1, 1, 1, 128]
    fully_connected_2 = tf.layers.conv2d(  # also really a fully connected layer, but without messing up the dimensions
        inputs = fully_connected_1,
        kernel_size=[1,1],
        strides=[1,1],
        filters=2,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        padding='VALID'
    )  # [1, 1, 1, 128] -> [1, 1, 1, 2]
    print(conv1.shape, mp_1.shape, conv2.shape, mp_2.shape, conv3.shape, mp_3.shape, fully_connected_1.shape, fully_connected_2.shape)
    return fully_connected_2

if __name__=='__main__':
    test_placeholder = tf.placeholder(shape=[1, 68, 68, 3], dtype=tf.float32)
    test_output_tensor = simple_dl_classifier(test_placeholder)
    print(test_output_tensor.shape)