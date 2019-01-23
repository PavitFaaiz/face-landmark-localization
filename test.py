import numpy as np
import utils
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import models

#Set to use only GPU#2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define constant
NUM_LANDMARKS = 76
weight_load_path = "weights.h5"

if __name__ == "__main__":
    #Config GPU options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    #Get data
    _, _, test_samples, test_targets = utils.get_data()
    image_shape = test_samples[0].shape
    #Get model
    model = models.get_model(image_shape)
    model.load_weights(weight_load_path)

    #Test the model on 10 random test data
    fig, ax = plt.subplots(2, 5)
    axs = ax.ravel()
    indices = np.arange(len(test_samples), dtype=np.int16)
    np.random.shuffle(indices)
    for idx, j in enumerate(list(indices[:10])):
        points = model.predict(test_samples[j:j+1])[0]
        points = np.reshape(points, [NUM_LANDMARKS, 2])
        points_target = test_targets[j:j+1,][0]
        points_target = np.reshape(points_target, [NUM_LANDMARKS, 2])
        axs[idx].set_aspect('equal')
        axs[idx].imshow(test_samples[j])
        for i in range(NUM_LANDMARKS):
            #Preprocess the points
            point = list(points[i])
            point_target = list(points_target[i])
            point[0] = point[0]*image_shape[1]
            point[1] = point[1]*image_shape[0]
            point_target[0] = point_target[0] * image_shape[1]
            point_target[1] = point_target[1] * image_shape[0]

            #Plot
            circle1 = plt.Circle(point, 2, color='r', linewidth=1) #Predicted points in red
            circle2 = plt.Circle(point_target, 2, color='g', linewidth=1) #Target points in green
            axs[idx].add_patch(circle1)
            axs[idx].add_patch(circle2)
    plt.show()
