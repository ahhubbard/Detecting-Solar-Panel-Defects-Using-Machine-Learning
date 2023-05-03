# TensorFlow and tf.keras
import tensorflow as tf
# The statement below was needed due to how Tensorflow is handled by PyCharm
import keras.api._v2.keras as keras
from keras.models import load_model

# Helper libraries
import os
import numpy as np

# Set to True to show print statements for the accuracy tests of each image
debug = True

try:
    home_path = os.getcwd()
    # print(home_path)
except:
    # This will be changed later to ask for the path
    print("Cannot find home path, no path finder implemented yet.\n"
          "Make sure that you use 'import os' in the Python file")

# Dictionary to define which folder to find the relevant test categories
PredicDic = {
    "Broken": home_path + "/Training_SolarImages/Predict/Broken/",
    "Cracked": home_path + "/Training_SolarImages/Predict/Cracked/",
    "Good": home_path + "/Training_SolarImages/Predict/Good/",
    "Hot": home_path + "/Training_SolarImages/Predict/Hot/",
    "Dirty": home_path + "/Training_SolarImages/Predict/Dirty/"
}


def Model_Accuracy_Test(image_dir):
    global debug
    image_size = (512, 512)
    num_classes = 5  # replace with the actual number of classes

    # load the model
    model = load_model(home_path + "/EPRIFeb28V1.h5")

    # define the class labels
    class_labels = ['Broken', 'Cracked', 'Bad', 'Good', 'Hot']  # replace with the actual class labels

    # initialize variables for calculating accuracy
    num_correct = 0
    num_total = 0
    # loop through each image in the folder
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # load and preprocess the image
            img_path = os.path.join(image_dir, filename)
            x = tf.keras.utils.load_img(img_path, target_size=image_size)
            x = tf.keras.utils.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # make the prediction
            probs = model.predict(x)[0]

            # get the predicted class label and probability
            pred_class_index = np.argmax(probs)
            pred_class_label = class_labels[pred_class_index]
            pred_prob = probs[pred_class_index] * 100.0

            # get the true class label from the filename
            true_class_label = os.path.splitext(filename)[0]

            # Print the predicted and true class labels and probabilities as live readouts
            if debug == True:
                print("Filename:", filename)
                print("Predicted class:", pred_class_label)
                print("Predicted probability:", pred_prob)
                print("True class:", true_class_label)

            # update the variables for calculating accuracy
            if pred_class_label == true_class_label:
                num_correct += 1
            num_total += 1

    # print the accuracy
    accuracy = num_correct / num_total
    print("Accuracy:", accuracy)
    return accuracy



broken_accuracy = Model_Accuracy_Test(PredicDic["Broken"])
good_accuracy = Model_Accuracy_Test(PredicDic["Good"])
crack_accuracy = Model_Accuracy_Test(PredicDic["Cracked"])
dirty_accuracy = Model_Accuracy_Test(PredicDic["Dirty"])
hot_accuracy = Model_Accuracy_Test(PredicDic["Hot"])

print("Broken Accuracy: ", broken_accuracy)
print("Good Accuracy: ", good_accuracy)
print("Microcrack Accuracy: ", crack_accuracy)
print("Dirty Accuracy: ", dirty_accuracy)
print("Hot Spot Accuracy: ", hot_accuracy)

