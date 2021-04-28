# %% [markdown]
# # Import pre-trained model from Keras
# Import a model pre-trained on ImageNet dataset. The parameter include_top=False
# excludes the final classification layer for ImageNet - which we will replace.

# %%
from keras import models
from keras import layers
from keras import preprocessing
import tensorflow as tf

from math import floor
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2

pre_model = InceptionResNetV2(
    weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# pre_model.summary()


# %% [markdown]
# # Visualizing first layer filters
# Normalize the weights of the filters used in the first convolution layer,
# then plot all 3 channels in color.
#
# The filters learned in the first layer are small and simple feature detectors.
# Correlations between these simple features are learned in subsequent
# layers of the network. Subsequent representations are composed of simpler features.
#
# It appears that the filters in the first layer consist of horizontal and vertical
# edge detectors, as well as blob detectors. The filters appear to be sensitive to color.


# Get weights for first convolutional layer
for layer in pre_model.layers:
    if 'conv' not in layer.name:
        continue

    filters = layer.get_weights()[0]
    break

# Normalize the values
fmin, fmax = filters.min(), filters.max()
filters = (filters - fmin) / (fmax - fmin)

# Plot filters
nf = filters.shape[3]
plt_w = 8
fig, axs = plt.subplots(int(nf/plt_w), plt_w)

for i in range(nf):
    f = filters[:, :, :, i]
    r, c = floor(i / plt_w), i % plt_w
    axs[r, c].imshow(f)
    axs[r, c].axis('off')

fig.suptitle('Convolution Layer 1 Filters')
plt.show()

# %% [markdown]
# # Load & Pre-Process Images
# Some pre-processing options:
# - Gaussian blurring (to smooth out noise)
# - Histogram leveling (does this work for 3-channel rgb?)
# - Normalizing image pixel intensities (divide all pixels by max value 255)
#
# keras.preprocessing.image_dataset_from_directory() is RBG by default, and
# resizes the images to (150, 150).
# TODO: Should label_mode='binary'? (would need binary_crossentropy loss)

dataset = preprocessing.image_dataset_from_directory(
    'dataset/training_set', image_size=(150, 150))

# Rescale/normalize image intensities by max value


def preprocess(images, labels):
    images = tf.cast(images/255., tf.float32)
    return images, labels


dataset = dataset.map(preprocess)

print(dataset.element_spec)

# Normalize pixel intensities to be in 0...1
dataset = dataset.map(lambda x, y: x/255)
