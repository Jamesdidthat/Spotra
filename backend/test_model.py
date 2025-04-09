import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os

# Define constants
IMG_SIZE = (224, 224)

# Load trained model
model = load_model("movie_recognition_model.h5")

# Get class names from dataset directory
class_names = sorted([d for d in os.listdir("dataset/") if os.path.isdir(os.path.join("dataset/", d))])

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def test_single_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")

test_single_image("dataset/interstellar/frame_150.jpg")  # Test an Interstellar image
test_single_image("dataset/joker/frame_150.jpg")         # Test a Joker image
test_single_image("dataset/inside_out/frame_150.jpg")    # Test an Inside Out image
