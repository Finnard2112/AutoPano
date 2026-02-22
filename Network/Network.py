"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image (128x128x2)
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    # inputs = layers.Input(shape=(128, 128, 2))

    # Conv Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(Img)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x) # Output: 64x64

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x) # Output: 32x32

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x) # Output: 16x16

    # Conv Block 4
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    prLogits = layers.Dense(8, activation=None)(x) # The raw numbers
    prSoftMax = layers.Activation('softmax')(prLogits) # Softmax applied to THOSE raw numbers
    return prLogits, prSoftMax

    # if mode == 'regression':
    #     # 8 real-valued offsets (4 points * 2 coords)
    #     outputs = layers.Dense(8, activation=None)(x)
    # elif mode == 'classification':
    #     # 168 outputs (8 variables * 21 quantization bins)
    #     outputs = layers.Dense(168, activation='softmax')(x)
    # else:
    #     raise ValueError("Mode must be 'regression' or 'classification'")

    # # model = models.Model(inputs=inputs, outputs=outputs, name=f"HomographyNet_{mode}")
    # return outputs

    # return H4Pt

