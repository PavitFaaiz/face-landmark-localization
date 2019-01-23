from keras import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
import keras
import tensorflow as tf

# Define constant
NUM_LANDMARKS = 76

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

def get_model(IMAGE_SHAPE):
    with tf.device("/cpu:0"):
        # Define model
        X = Input(shape=IMAGE_SHAPE)
        baseModel = ResNet50(include_top=False)
        pooled = GlobalAveragePooling2D()(baseModel(X))
        dense = Dense(2 * NUM_LANDMARKS, activation="sigmoid", kernel_initializer="glorot_uniform")
        out = dense(pooled)
        model = Model(inputs=X, outputs=out)
    return model