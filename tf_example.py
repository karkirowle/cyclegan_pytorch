import tensorflow as tf

inputs = tf.ones([10,24,128,1])

filters = 128
kernel_size = (3,3)
strides = (1,2)
padding = "same"
kernel_initializer = None
activation = None
name = "sajt"

conv_layer_1 = tf.layers.conv2d(
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding)


filters = 256
kernel_size = (3,3)
strides = (2,2)

print(conv_layer_1.shape)


conv_layer_2 = tf.layers.conv2d(
    inputs=conv_layer_1,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding)
kernel_size = (3,3)
strides = (2,2)
filters = 512
print(conv_layer_2.shape)

conv_layer_3 = tf.layers.conv2d(
    inputs=conv_layer_2,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding)

filters = 1024
kernel_size = (6,3)
strides = (1,2)

print(conv_layer_3.shape)


conv_layer_4 = tf.layers.conv2d(
    inputs=conv_layer_3,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding)

print(conv_layer_4.shape)