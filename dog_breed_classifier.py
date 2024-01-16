# Import necessary libraries
from taipy.gui import Gui
from logic import preprocessing_image, extract_unique_breeds_info, predict_image
import numpy as np

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
        top_pred, top_prob = predict_image(var_value, unique_breeds)
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
