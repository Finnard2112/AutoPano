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
    conv1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_1')(Img)
    conv1_1 = tf.keras.layers.BatchNormalization(name='bn1_1')(conv1_1)
    conv1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = tf.keras.layers.BatchNormalization(name='bn1_2')(conv1_2)
    pool1 = tf.keras.layers.MaxPooling2D(2, 2, name='pool1')(conv1_2)

    # Conv Block 2
    conv2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_1 = tf.keras.layers.BatchNormalization(name='bn2_1')(conv2_1)
    conv2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = tf.keras.layers.BatchNormalization(name='bn2_2')(conv2_2)
    pool2 = tf.keras.layers.MaxPooling2D(2, 2, name='pool2')(conv2_2)

    # Conv Block 3
    conv3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_1 = tf.keras.layers.BatchNormalization(name='bn3_1')(conv3_1)
    conv3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = tf.keras.layers.BatchNormalization(name='bn3_2')(conv3_2)
    pool3 = tf.keras.layers.MaxPooling2D(2, 2, name='pool3')(conv3_2)

    # Conv Block 4
    conv4_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_1 = tf.keras.layers.BatchNormalization(name='bn4_1')(conv4_1)
    conv4_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_2 = tf.keras.layers.BatchNormalization(name='bn4_2')(conv4_2)

    # Fully Connected Layers
    flat = tf.keras.layers.Flatten(name='flatten')(conv4_2)
    fc1 = tf.keras.layers.Dense(1024, activation='relu', name='fc1')(flat)
    drop1 = tf.keras.layers.Dropout(0.5, name='dropout1')(fc1)
    fc2 = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(drop1)
    drop2 = tf.keras.layers.Dropout(0.5, name='dropout2')(fc2)

    # Output Layer (8 homography values)
    prLogits = tf.keras.layers.Dense(8, activation=None, name='output')(drop2)
    prSoftMax = prLogits
    
    return prLogits, prSoftMax

