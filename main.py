from keras.applications.resnet50 import ResNet50
import keras
from keras import Model, Input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pandas as pd
import numpy as np
import utils
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model

#Set to use only GPU#2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define constant
NUM_LANDMARKS = 76
IMAGE_SHAPE = [640//2, 480//2, 3]

#Define a loss function where target matrix might have missing values
def mse_with_dontcare(T, Y):
    ##Find the pairs in T having x-y coordinate = (0, 0)
    #Reshape to have x-y coordinate dimension
    T_reshaped = tf.reshape(T, shape=[-1, 2, NUM_LANDMARKS])
    Y_reshaped = tf.reshape(Y, shape=[-1, 2, NUM_LANDMARKS])
    #Calculate Euclidean distance between each pair of points
    distance = tf.norm(T_reshaped - Y_reshaped, ord='euclidean', axis=1)

    #If summing x and y yields zero, then (x, y) == (0, 0)
    T_summed = tf.reduce_sum(T_reshaped, axis=1)
    zero = tf.constant(0.0)
    mask = tf.not_equal(T_summed, zero)
    #Get the interested samples and calculate loss with Mean of Euclidean distance
    masked_distance = tf.boolean_mask(distance, mask)
    return tf.reduce_mean(masked_distance)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    with tf.device("/cpu:0"):
        ##Get and process data
        samples, annotations = utils.load_data(os.getcwd(), IMAGE_SHAPE)
        samples = samples.astype(np.float16)
        annotations = annotations.astype(np.float32)
        samples = samples/255 #Normalize the images
        annotations = annotations/2 #As we shrink the size by 2x2
        annotations = np.reshape(annotations, [len(annotations), NUM_LANDMARKS, 2])
        annotations[:, :, 0] /= IMAGE_SHAPE[1]
        annotations[:, :, 1] /= IMAGE_SHAPE[0]
        annotations = np.reshape(annotations, [len(annotations), NUM_LANDMARKS * 2])

        #Separate into training and testing sets
        train_portion = 0.7
        num_train = int(len(samples) * 0.7)
        num_test = len(samples) - num_train
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        train_samples = samples[indices[0:num_train]]
        train_targets = annotations[indices[0:num_train]]
        test_samples = samples[indices[num_train:]]
        test_targets = annotations[indices[num_train:]]

        #Define model
        X = Input(shape=IMAGE_SHAPE)
        baseModel = ResNet50(include_top=False)
        pooled = GlobalAveragePooling2D()(baseModel(X))
        dense = Dense(2 * NUM_LANDMARKS, activation="sigmoid", kernel_initializer="glorot_uniform")
        out = dense(pooled)
        model = Model(inputs=X, outputs=out)
        model.compile(keras.optimizers.Adam(lr=0.00025), loss=mse_with_dontcare)
    model.load_weights("weights.h5")
    model.fit(train_samples, train_targets, validation_data=(test_samples, test_targets),
              epochs=1, batch_size=16)

    #Test the model on 10 random test data
    fig, ax = plt.subplots(2, 5)
    axs = ax.ravel()
    for j in range(10):
        points = model.predict(test_samples[j:j+1])[0]
        points = np.reshape(points, [NUM_LANDMARKS, 2])
        points_target = test_targets[j:j+1,][0]
        points_target = np.reshape(points_target, [NUM_LANDMARKS, 2])
        axs[j].set_aspect('equal')
        axs[j].imshow(test_samples[j])
        for i in range(NUM_LANDMARKS):
            #Preprocess the points
            point = list(points[i])
            point_target = list(points_target[i])
            point[0] = point[0]*IMAGE_SHAPE[1]
            point[1] = point[1]*IMAGE_SHAPE[0]
            point_target[0] = point_target[0] * IMAGE_SHAPE[1]
            point_target[1] = point_target[1] * IMAGE_SHAPE[0]

            #Plot
            circle1 = plt.Circle(point, 2, color='r', linewidth=1)
            circle2 = plt.Circle(point_target, 2, color='g', linewidth=1)
            axs[j].add_patch(circle1)
            axs[j].add_patch(circle2)
    plt.show()
