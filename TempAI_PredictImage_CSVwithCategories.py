# TensorFlow and tf.keras
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.models import load_model


# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

image_size = (150, 150)
img_path = "C:/Users/Adam/PycharmProjects/EPRIProject/Training_SolarImages/Predict/Good/Good_1341.png"

model = load_model('C:/Users/Adam/PycharmProjects/EPRIProject/EPRIFeb28V1.h5')

img = tf.keras.utils.load_img(img_path, target_size=image_size)
# plt.imshow(img)
# plt.show()

# load and preprocess the image
x = tf.keras.utils.load_img(img_path, target_size=image_size)
x = tf.keras.utils.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# define the class labels
class_labels = ['Broken', 'Microcracks', 'Dirty', 'Good', 'Hot Spots']

# define a custom decode_predictions function
def my_decode_predictions(preds, top=5):
    results = []
    for pred in preds:
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        top_indices = pred.argsort()[:, -top:][:, ::-1]
        result = []
        for i in range(pred.shape[0]):
            result.append([(class_labels[j], pred[i, j]*100.0) for j in top_indices[i]])
        results.append(result)
    return results[0]

import os
import csv

# Set directory path
dir_path = "C:/Users/Adam/PycharmProjects/EPRIProject/Training_SolarImages/Predict/Dirty/"

# Set image size
image_size = (150, 150)

# Define output file path
output_path = "predictionsdirty.csv"

# Open output file
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)

    # Iterate over all image files in directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.png'):
            img_path = os.path.join(dir_path, file_name)

            # Load and preprocess image
            img = tf.keras.utils.load_img(img_path, target_size=image_size)
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # Make prediction
            probs = model.predict(x)[0]
            decoded_probs = my_decode_predictions(probs.reshape(1, len(class_labels)), top=5)

            # Write prediction to output file
            writer.writerow([file_name] + [f"{label}: {prob:.2f}%" for label, prob in decoded_probs[0]])

print("Predictions saved to:", output_path)


# # make the prediction
# probs = model.predict(x)[0]
#
# num_classes = 5 # replace with the actual number of classes
# decoded_probs = my_decode_predictions(probs.reshape(1, num_classes), top=5)
#
#
# # print the predicted class labels and probabilities
# for i in range(len(decoded_probs)):
#     print(f"Image {i+1}:")
#     for j in range(len(decoded_probs[i])):
#         print(f"{decoded_probs[i][j][0]}: {decoded_probs[i][j][1]:.2f}%")


