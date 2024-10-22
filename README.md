# Dog Breed Classifier

## Overview

This Python script utilizes TensorFlow and Keras to create a simple dog breed classifier with a graphical user interface (GUI) provided by the Taipy library. The classifier is based on a pre-trained model using TensorFlow Hub and ResNet V2.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Taipy
- TensorFlow
- Keras
- TensorFlow Hub
- NumPy
- pandas

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Daki-15/dog-breed-classifier.git
```

2. Navigate to the project directory:

```bash
cd dog-breed-classifier
```

3. Download the pre-trained model and place it in the project directory. The model should be a .h5 file. (https://drive.google.com/file/d/1-vIbOU6BfO6ZlM10xaFu2eAyN73pl1qb/view?usp=drive_link)

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Run the script:

```bash
python dog_breed_classifier.py
```

The script will launch a graphical user interface where you can select an image of a dog from your file system. The model will predict the breed of the dog based on the pre-trained classifier.

## File Structure

- **dog_breed_classifier.py:** The main Python script containing the classifier logic and GUI setup.
- **./img/logo.png:** Logo image used in the GUI.
- **./img/placeholder_image.png:** Placeholder image displayed initially.

## Configuration

- **model_path:** Path to the pre-trained model file. (The `.h5` file was not added to github due to the site's 100MB limit) 
- **labels.csv:** CSV file containing dog breed labels.
- **index:** HTML template for the GUI interface.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## App interface:

![](./img/interface.PNG)

### Demo:

German Shepherd 
![](./img/Example_1.PNG)
French bulldog
![](./img/Example_2.PNG)
