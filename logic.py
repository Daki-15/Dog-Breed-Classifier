# Import necessary libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
tf.get_logger().setLevel(logging.ERROR)

# Define the path to the pre-trained model file
model_path = "model.h5"

# Load the pre-trained model using Keras and TensorFlow Hub
model = load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

def preprocessing_image(image_path):
    """
    Takes an image file path and turns the image into a Tensor.
    """
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Decode the JPEG image into a numerical Tensor with 3 color channels (RGB - Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Normalize the image
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
    # Resize the image to the desired size (224, 224)
    image = tf.image.resize(image, size=(224, 224))

    return image

def extract_unique_breeds_info(race_info_path):
    # Read the CSV file containing unique breeds and their information
    race_info_csv = pd.read_csv(race_info_path)
    
    # Extract the "breed" column from the CSV and convert it to a NumPy array
    breeds = np.array(race_info_csv["breed"])
    
    # Extract the "interesting things" column from the CSV and convert it to a NumPy array
    interesting_things = np.array(race_info_csv["interesting"])

    return breeds, interesting_things

def predict_image(path_to_img, unique_breeds, model=model):
    # Preprocess the image using the defined function
    img = preprocessing_image(path_to_img)
    # Predict the image using the loaded model
    prob = model.predict(np.array([img])[:1])
    # Find the highest probability and the corresponding predicted breed
    top_prob = prob.max()
    top_pred = unique_breeds[np.argmax(prob)]

    return top_pred, top_prob
