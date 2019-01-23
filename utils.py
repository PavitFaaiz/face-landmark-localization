from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import pickle

def load_data(root_dir, image_shape):
    if not os.path.exists(os.getcwd()+"\\data.p"):
        image_path = root_dir + "images\\"
        annotate_path = root_dir + "muct76-opencv.csv"

        #Process samples
        filenames = os.listdir(image_path)
        data = np.zeros([len(filenames)]+image_shape, dtype=np.uint8)
        for idx, f in enumerate(filenames):
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

def resize_images(path, new_size):
    import cv2
    filenames = os.listdir(path)
    for f in filenames:
        img = cv2.imread(path+f)
        img = cv2.resize(img, new_size)
        cv2.imwrite(path+f, img)
