import cv2
import numpy as np
import os
import sys
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow.python.keras.backend import sign

IMG_WIDTH = 30
IMG_HEIGHT = 30

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    data_dir=sys.argv[1]
    #path=os.path.join(data_dir,"saved_model")
    # print(path)
    filePath=os.path.join("",)
    fileList=os.listdir("signs")
    print(fileList)
    # img=cv2.imread("signs/test.ppm")
    # new=cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    # print(img)
    image = tf.keras.preprocessing.image.load_img("signs/test.ppm", target_size=(IMG_WIDTH,IMG_HEIGHT))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    loaded_model = tf.keras.models.load_model(data_dir)
    
    pred=loaded_model.predict(input_arr)
    print(pred)


if __name__ == "__main__":
    main()