import numpy as np
import tensorflow as tf


def cv_norm(img):
    im = img / 255.0
    return im.astype(np.float32)


def cv_inv_norm(img):
    img_rgb = img * 255.0
    return img_rgb.astype(np.float32)

    
def lrelu(x, leak=0.2,):
    return tf.maximum(x, leak * x)


def make_var(name, shape, trainable=True, initializer=tf.constant_initializer(0.0)):
    return tf.get_variable(name, shape, trainable=trainable, initializer=initializer)


def relu(input_, name="relu"):
    return tf.nn.relu(input_, name=name)


def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=True):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim], initializer=tf.constant_initializer(0.0))
            output = tf.nn.bias_add(output, biases)
        return output


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

