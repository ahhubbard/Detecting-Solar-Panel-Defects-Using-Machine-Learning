import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Load the model from the h5 file
model = load_model('EPRIFeb28V1.h5')
img_width, img_height = 512, 512

# Load an image to visualize the feature maps
img_path = '/Training_SolarImages/Predict/Broken/Broken0.jpg'
img = load_img(img_path, target_size=(img_width, img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Define a new model that outputs the feature maps for all layers
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

# Generate the feature maps for the input image
successive_feature_maps = visualization_model.predict(x)

# Retrieve the names of the layers
layer_names = [layer.name for layer in model.layers]

# Plot the feature maps for the conv/maxpool layers, not the fully connected layers
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if 'conv2d' not in layer_name and 'max_pooling2d' not in layer_name:
        continue

    n_features = feature_map.shape[-1]  # number of features in the feature map
    size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))

    # Postprocess the feature maps to be visually palatable
    for i in range(n_features):
        x = feature_map[0, :, :, i]
        x -= x.mean()
        x /= x.std()
        x *= 64
        x += 128
        x = np.clip(x, 0, 255).astype('uint8')

        # Tile each filter into a horizontal grid
        display_grid[:, i * size:(i + 1) * size] = x

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    scale = 10. / size  # fixed height of 10 inches
    fig = plt.figure(figsize=(scale * n_features, scale))
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1.0)
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, cmap='viridis')

    # plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
