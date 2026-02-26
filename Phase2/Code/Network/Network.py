import os
# MANDATORY: This must happen BEFORE any tensorflow import
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize, is_training):
    # Enforce static shape for the graph
    Img.set_shape([MiniBatchSize, 128, 128, 6])
    
    # We use tf.layers (V1 style) which is compatible with tf-keras legacy mode
    def conv_bn_relu(x, filters, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Using the direct layers interface
            x = tf.layers.conv2d(x, filters, 3, padding='same', name='conv')
            x = tf.layers.batch_normalization(x, training=is_training, name='bn')
            x = tf.nn.relu(x)
        return x

    # Block 1
    x = conv_bn_relu(Img, 64, 'b1_1')
    x = conv_bn_relu(x, 64, 'b1_2')
    x = tf.layers.max_pooling2d(x, 2, 2, name='p1')

    # Block 2
    x = conv_bn_relu(x, 64, 'b2_1')
    x = conv_bn_relu(x, 64, 'b2_2')
    x = tf.layers.max_pooling2d(x, 2, 2, name='p2')

    # Block 3
    x = conv_bn_relu(x, 128, 'b3_1')
    x = conv_bn_relu(x, 128, 'b3_2')
    x = tf.layers.max_pooling2d(x, 2, 2, name='p3')

    # Block 4
    x = conv_bn_relu(x, 128, 'b4_1')
    x = conv_bn_relu(x, 128, 'b4_2')

    # Fully Connected
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='fc1')
    x = tf.layers.dropout(x, rate=0.5, training=is_training, name='d1')
    
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='fc2')
    x = tf.layers.dropout(x, rate=0.5, training=is_training, name='d2')

    # Output: tanh for [-1, 1] range (Common for Homography Net delta-four-point)
    prLogits = tf.layers.dense(x, 8, activation=tf.nn.tanh, name='out')
    prSoftMax = prLogits # Regression doesn't use Softmax, but kept for your return signature
    
    return prLogits, prSoftMax