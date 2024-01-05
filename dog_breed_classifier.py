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

def predict_image(model, path_to_img, unique_breeds):
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
unique_breeds, interesting_things = extract_unique_breeds_info("unique_breeds.csv")
prob = 0
pred = ""
interesting_placeholder = ""

# Define the HTML template for the user interface
index = """
<|text-center|
<|{"./img/logo.png"}|image||width=25vw|>

<|{content}|file_selector|extensions=.jpg|>
Select an image of a dog from your file system

<|{pred}|>


<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

<|{interesting_placeholder}|>
>
"""


def on_change(state, var_name, var_value):
    # Handle changes in the GUI elements, particularly when a new image or breed is selected
    if var_name == "content":
        top_pred, top_prob = predict_image(model, var_value, unique_breeds)
        state.prob = round(top_prob * 100)
        state.pred = "This is a \n\n" + top_pred
        state.image_path = var_value
        
        # Set the interesting information based on the selected breed
        breed_index = np.where(unique_breeds == top_pred)[0][0]
        interesting_info = interesting_things[breed_index]
        state.interesting_placeholder = f"{interesting_info}"

# Create an instance of the GUI
app = Gui(page=index)

# Run the application with reloader enabled
if __name__ == "__main__":
    app.run(use_reloader=True)
