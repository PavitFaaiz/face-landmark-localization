import models
import keras
import utils

weight_load_path = None
weight_save_path = "weights.h5"
optimizer = keras.optimizers.Adam
lr = 0.001
epochs = 5
batch_size = 16

if __name__ == "__main__":
    # Get data
    train_samples, train_targets, test_samples, test_targets = utils.get_data()
    image_shape = train_samples[0].shape
    model = models.getModel(image_shape)
    model.compile(keras.optimizer(lr=lr), loss=model.mse_with_dontcare)
    if weight_load_path is not None:
        model.load_weights(weight_load_path)
    #Begin training
    model.fit(train_samples, train_targets,
              epochs=epochs, batch_size=batch_size,
              validation_data=(test_samples, test_targets))
    model.save_weights(weight_save_path)

