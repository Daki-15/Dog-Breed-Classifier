# Import necessary libraries
from taipy.gui import Gui
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd

# Define the path to the pre-trained model file
model_path = "10-12-23--11_51_1702209080-10222-images-resnet_v2-Adam.h5"

# Load the pre-trained model using Keras and TensorFlow Hub
model = load_model(model_path,
                    custom_objects={"KerasLayer" : hub.KerasLayer})

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

def extract_unique_breeds(labels_path):
    # Read the CSV file containing labels
    labels_csv = pd.read_csv(labels_path)

    # Extract the "breed" column from the CSV and convert it to a NumPy array
    labels = labels_csv["breed"]
    labels = np.array(labels)

    return np.unique(labels)

def predict_image(model, path_to_img):
    # Preprocess the image using the defined function
    img = preprocessing_image(path_to_img)
    # Predict the image using the loaded model
    prob = model.predict(np.array([img])[:1])
    # Find the highest probability and the corresponding predicted breed
    top_prob = prob.max()
    top_pred = unique_breeds[np.argmax(prob)]

    return top_pred, top_prob

# Initialize variables
content = ""
image_path = "./img/placeholder_image.png"
unique_breeds = extract_unique_breeds("labels.csv")
prob = 0
pred = ""

# Define the HTML template for the user interface
index = """
<|text-center|
<|{"./img/logo.png"}|image||width=25vw|>

<|{content}|file_selector|extensions=.jpg|>
Select an image of a dog from your file system

<|{pred}|>

<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_value):
    # Handle changes in the GUI elements, particularly when a new image is selected
    if var_name == "content":
        top_pred, top_prob = predict_image(model, var_value)
        state.prob = round(top_prob * 100)
        state.pred = "This is a \n" + top_pred
        state.image_path = var_value

# Create an instance of the GUI
app = Gui(page=index)

# Run the application with reloader enabled
if __name__ == "__main__":
    app.run(use_reloader=True)
