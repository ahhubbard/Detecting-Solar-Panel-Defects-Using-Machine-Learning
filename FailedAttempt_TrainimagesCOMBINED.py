# Import necessary modules
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

try:
    home_path = os.getcwd()
    # print(home_path)
except:
    # This will be changed later to ask for the path
    print("Cannot find home path, no path finder implemented yet.\n"
          "Make sure that you use 'import os' in the Python file")

# Path to your dataset
data_dir = home_path + "/Training_SolarImages/Data/"
train_data_dir = home_path + "/Training_SolarImages/Train/"
validation_data_dir = home_path + "/Training_SolarImages/Validate/"

# List of your class names
class_names = ['Broken', 'Cracked', 'Good', 'Hot', 'Dirty']

# Number of classes
num_classes = len(class_names)

# Define image size and batch size
img_width, img_height = 512, 512
batch_size = 32

# Load the original data into memory
data = []
labels = []
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))
            data.append(img)
            labels.append(class_name)
            print(f"{img_path}")
        except Exception as e:
            print(f"Could not load {img_path}: {e}")

# Define an image generator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=5,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    cval=0,
    brightness_range=[0.5, 1.5],
    channel_shift_range=80.0
)

# Create the augmented data and split into train and validation sets
train_data = []
train_labels = []
val_data = []
val_labels = []
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    class_data = []
    print(class_data)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))
        class_data.append(img)
    train_data_class, val_data_class, train_labels_class, val_labels_class = train_test_split(class_data,
                                                                                              [class_name] * len(
                                                                                                  class_data),
                                                                                              test_size=0.2,
                                                                                              random_state=42)
    train_data.extend(train_data_class)
    train_labels.extend(train_labels_class)
    val_data.extend(val_data_class)
    val_labels.extend(val_labels_class)


# Convert your data and labels to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)
print(train_data)
print(val_data)
# Rescale your data
train_data = train_data.astype('float32') / 255
val_data = val_data.astype('float32') / 255

# # Convert labels to one-hot encoding
# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
# val_labels = tf.keras.utils.to_categorical(val_labels, num_classes)

from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
le = LabelEncoder()

# Fit the encoder on the label data
le.fit(train_labels)

# Encode the labels as integers
train_labels = le.transform(train_labels)
val_labels = le.transform(val_labels)

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, len(class_names))
val_labels = tf.keras.utils.to_categorical(val_labels, len(class_names))


# Create an image generator with additional augmentation options for training data
train_datagen = ImageDataGenerator(
    rotation_range=5,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    cval=0,
    brightness_range=[0.5, 1.5],
    channel_shift_range=80.0
)

# Create an ImageDataGenerator object for validation data
val_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
)

# Set batch size and number of epochs
batch_size = 16
epochs = 50

# Create a generator for the training data
train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)

# Create a generator for the validation data
val_generator = val_datagen.flow(val_data, val_labels, batch_size=batch_size)

input_shape = (img_width, img_height, 3)

# Define your model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the model using the generators
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_data) // batch_size
)

# Save the model
model.save('EPRIFApril2_V1_Attempt1.h5')
model.save_weights('EPRIApril5V1_Attempt1.h5')

