"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image (128x128x6)
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    
    # Conv Block 1
    conv1_1 = tf.layers.conv2d(Img, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_1')
    conv1_1 = tf.layers.batch_normalization(conv1_1, training=True, name='bn1_1')
    conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_2')
    conv1_2 = tf.layers.batch_normalization(conv1_2, training=True, name='bn1_2')
    pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=2, strides=2, name='pool1')

    # Conv Block 2
    conv2_1 = tf.layers.conv2d(pool1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_1')
    conv2_1 = tf.layers.batch_normalization(conv2_1, training=True, name='bn2_1')
    conv2_2 = tf.layers.conv2d(conv2_1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_2')
    conv2_2 = tf.layers.batch_normalization(conv2_2, training=True, name='bn2_2')
    pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, name='pool2')

    # Conv Block 3
    conv3_1 = tf.layers.conv2d(pool2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_1')
    conv3_1 = tf.layers.batch_normalization(conv3_1, training=True, name='bn3_1')
    conv3_2 = tf.layers.conv2d(conv3_1, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_2')
    conv3_2 = tf.layers.batch_normalization(conv3_2, training=True, name='bn3_2')
    pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=2, strides=2, name='pool3')

    # Conv Block 4
    conv4_1 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_1')
    conv4_1 = tf.layers.batch_normalization(conv4_1, training=True, name='bn4_1')
    conv4_2 = tf.layers.conv2d(conv4_1, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_2')
    conv4_2 = tf.layers.batch_normalization(conv4_2, training=True, name='bn4_2')

    # Fully Connected Layers
    flat = tf.layers.flatten(conv4_2, name='flatten')
    fc1 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu, name='fc1')
    drop1 = tf.layers.dropout(fc1, rate=0.5, training=True, name='dropout1')
    fc2 = tf.layers.dense(drop1, units=1024, activation=tf.nn.relu, name='fc2')
    drop2 = tf.layers.dropout(fc2, rate=0.5, training=True, name='dropout2')

    # Output Layer (8 homography values)
    prLogits = tf.layers.dense(drop2, units=8, activation=None, name='output')
    prSoftMax = prLogits
    
    return prLogits, prSoftMax

