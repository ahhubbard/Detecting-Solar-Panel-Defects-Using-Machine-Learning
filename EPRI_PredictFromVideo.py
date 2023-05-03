# TensorFlow and tf.keras
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.models import load_model

# Helper libraries
import os
import cv2
import numpy as np

# Constants
MODEL = 'EPRIFeb28V1.h5'
image_size = (512, 512)
video_file = "C:/Users/Adam/PycharmProjects/EPRIProjectV2/Training_SolarImages/Data/RealTest/Good/Good_Test_95.mp4"
num_classes = 5 # replace with the actual number of classes

# Try to get directory path
try:
    # Get Directory path and check if it's valid
    home_path = os.getcwd()
    # load the model
    model = load_model(home_path + '/' + MODEL)

    # print(home_path)
except:
    # This will be changed later to ask for the path

    print(f"No model by the name of {MODEL} was found. ")

# define the class labels
class_labels = ['Broken', 'Microcracks', 'Bad', 'Good', 'Hot Spots'] # replace with the actual class labels

# Load the video and get the frames
cap = cv2.VideoCapture(video_file)
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
    else:
        break
cap.release()

# Loop through each frame in the video
predictions = []
for frame in frames:
    # Use object detection to detect the solar panel in the frame
    # (This code assumes you have implemented object detection and have the bounding box coordinates of the solar panel)
    x1, y1, x2, y2 = (0, 0, 100, 100) # replace with the actual bounding box coordinates of the solar panel
    solar_panel = frame[y1:y2, x1:x2]

    # Resize the solar panel to multiple resolutions
    resized_frames = [cv2.resize(solar_panel, (512, 512)), cv2.resize(solar_panel, (768, 768)), cv2.resize(solar_panel, (1024, 1024))]


    # Make predictions for each resized frame
    probs_list = []
    for resized_frame in resized_frames:
        # Preprocess the frame
        x = np.expand_dims(resized_frame, axis=0)
        x = tf.cast(x, tf.float32)
        if x.shape[1:] != image_size:
            x = tf.image.resize(x, image_size, method='nearest')

        x = x / 255.0

        # Make the prediction
        probs = model.predict(x)[0]

        # Add the prediction to the list
        probs_list.append(probs)

    # Average the predictions
    probs = np.mean(probs_list, axis=0)

    # Get the predicted class label and probability
    pred_class_index = np.argmax(probs)
    pred_class_label = class_labels[pred_class_index]
    pred_prob = probs
    # Add the prediction to the list of predictions
    predictions.append((pred_class_label, pred_prob))

# Calculate the average accuracy
num_correct = 0
num_total = len(predictions)

accuracy = num_correct / num_total
print("Accuracy:", accuracy)
