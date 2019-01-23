from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import pickle

# Define constant
NUM_LANDMARKS = 76

def load_data(root_dir):
    if not os.path.exists(os.getcwd()+"\\data.p"):
        image_path = root_dir + "images\\"
        annotate_path = root_dir + "muct76-opencv.csv"

        #Process samples
        filenames = os.listdir(image_path)
        img = image.load_img(image_path + filenames[0])
        x = image.img_to_array(img)
        filenames[0] = filenames[0][0:len(filenames[0]) - 4] #Remove extension in file names
        image_shape = x.shape
        data = np.zeros([len(filenames)]+image_shape, dtype=np.uint8)
        data[0] = x
        for idx, f in enumerate(filenames[1:]):
            img = image.load_img(image_path + f)
            x = image.img_to_array(img)
            filenames[idx] = f[0:len(f)-4]
            data[idx] = x

        ##Process annotations
        df = pd.read_csv(annotate_path, index_col=False)
        df = df.drop(["tag"], axis=1) #We don't need their identity, so, drop it!
        subset = df['name'].where(df['name'].isin(filenames))#Get only the rows of the images we have
        df = df[subset.notna()]
        df = df.drop("name", axis=1) #Drop the name column also
        pickle.dump((data, df.values), open("data.p", "wb")) #Save the data with pickle to be easily accessed later
    else:
        data, df = pickle.load(open("data.p", "rb"))
    return data, df

def get_data():
    ##Get and process data
    samples, annotations = load_data(os.getcwd())
    samples = samples.astype(np.float16)
    image_shape = samples[0].shape
    annotations = annotations.astype(np.float32)
    samples = samples / 255  # Normalize the images
    annotations = annotations / 2  # As we shrink the size by 2x2
    annotations = np.reshape(annotations, [len(annotations), NUM_LANDMARKS, 2])
    annotations[:, :, 0] /= image_shape[1]
    annotations[:, :, 1] /= image_shape[0]
    annotations = np.reshape(annotations, [len(annotations), NUM_LANDMARKS * 2])

    # Separate into training and testing sets
    train_portion = 0.7
    num_train = int(len(samples) * train_portion)
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    train_samples = samples[indices[0:num_train]]
    train_targets = annotations[indices[0:num_train]]
    test_samples = samples[indices[num_train:]]
    test_targets = annotations[indices[num_train:]]
    return train_samples, train_targets, test_samples, test_targets

def resize_images(path, new_size):
    import cv2
    filenames = os.listdir(path)
    for f in filenames:
        img = cv2.imread(path+f)
        img = cv2.resize(img, new_size)
        cv2.imwrite(path+f, img)
