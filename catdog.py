# %% [markdown]
#  # Import pre-trained model from Keras
#  Import a model pre-trained on ImageNet dataset. The parameter include_top=False
#  excludes the final classification layer for ImageNet - which we will replace.

# %%
from tensorflow.keras.applications import InceptionResNetV2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import preprocessing
from keras import layers
from keras import models
import pandas as pd
import seaborn as sn
sn.set_theme()


pre_model = InceptionResNetV2(
    weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# pre_model.summary()

# %% [markdown]
#  # Visualizing first layer filters
#  Normalize the weights of the filters used in the first convolution layer,
#  then plot all 3 channels in color.
#
#  The filters learned in the first layer are small and simple feature detectors.
#  Correlations between these simple features are learned in subsequent
#  layers of the network. Subsequent representations are composed of simpler features.
#
#  It appears that the filters in the first layer consist of horizontal and vertical
#  edge detectors, as well as blob detectors. The filters appear to be sensitive to color.
#  Get weights for first convolutional layer

# %%
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

for i, ax in enumerate(fig.axes):
    f = filters[:, :, :, i]
    ax.imshow(f)
    ax.set_axis_off()

fig.suptitle('Convolution Layer 1 Filters')
plt.show()

# %% [markdown]
#  # Load & Pre-Process Images
#  Some pre-processing options:
#  - Gaussian blurring (to smooth out noise)
#  - Histogram leveling (does this work for 3-channel rgb?)
#  - Normalizing image pixel intensities (divide all pixels by max value 255)
#
#  keras.preprocessing.image_dataset_from_directory() is RBG by default, and
#  resizes the images to (150, 150).
#  TODO: Should label_mode='binary'? (would need binary_crossentropy loss)

# %%
train_ds = preprocessing.image_dataset_from_directory(
    'dataset/training_set', image_size=(150, 150))

test_ds = preprocessing.image_dataset_from_directory(
    'dataset/test_set', image_size=(150, 150))

# Visualize some of the data
for images, labels in train_ds.take(1):
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(fig.axes):
        ax.imshow(images[i].numpy().astype('uint8'))
        ax.set_title(int(labels[i]))
        ax.set_axis_off()

# Rescale/normalize image intensities by max value


def preprocess(images, labels):
    images = tf.cast(images/255., tf.float32)
    return images, labels


train_ds = train_ds.map(preprocess)
# use buffered prefetch to yield data from disk without I/O blocking
train_ds = train_ds.prefetch(buffer_size=32)


# %%
model = models.Sequential()
model.add(pre_model)  # inception resnet v2
model.add(layers.Flatten())  # flattens output of pre_model
model.add(layers.Dense(256, activation='relu'))  # adds a dense layer
# final output layer w/ single neuron, output in range [0,1]
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
pre_model.trainable = False  # freeze pre-trained model weights


# %%
# Evaluate model against test & get predictions
model.compile(loss='binary_crossentropy')
evaluation = model.evaluate(test_ds, verbose=1)
print(f'Loss: {evaluation[0]}\tAccuracy: {evaluation[1]}')

predictions = model.predict(test_ds)

# %%

# Display predictions in confusion matrix


def binaryConfusionMatrix(X, Y):
    nclasses = 2
    cm = np.zeros((nclasses, nclasses))
    dat = np.concatenate((X, Y), axis=1)
    for res in dat:
        cm[int(res[0]), int(res[1])] += 1
    return cm


labels = np.concatenate([y for _, y in test_ds], axis=0).reshape(-1, 1)
assert predictions.shape == labels.shape
cm = binaryConfusionMatrix(predictions, labels)

df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True)
plt.show()


# %%
# history = model.fit(train_ds, validation_split=0.3, epochs=10)
