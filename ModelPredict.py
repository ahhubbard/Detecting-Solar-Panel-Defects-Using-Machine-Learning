import cv2
from urllib.request import urlopen
import numpy as np
from keras.models import load_model

# Load the Keras model
MODEL = 'C:/Users/Adam/PycharmProjects/EPRIProject/SympModel.h5'
model = load_model(MODEL)

# Define the class labels
class_labels = ['Broken', 'Good']

# Set up the OpenCV camera feed
url = r'http://192.168.1.2/capture'
cv2.namedWindow("Camera")

# Process incoming images
while True:
    # Read an image from the camera feed
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    img = cv2.imdecode(imgnp, -1)

    # Preprocess the image
    x = cv2.resize(img, (240, 240))
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Make the prediction
    probs = model.predict(x)[0]

    # Get the predicted class label and probability
    pred_class_index = np.argmax(probs)
    pred_class_label = class_labels[pred_class_index]
    pred_prob = probs[pred_class_index] * 100.0

    # Display the predicted label and probability on the image
    cv2.putText(img, f"{pred_class_label} ({pred_prob:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image and wait for a key press
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == ord('q'):
        break

    # Wait a few seconds before closing the window and getting another image
    cv2.waitKey(1000)

# Clean up
cv2.destroyAllWindows()
