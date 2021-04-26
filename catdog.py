# %% [markdown]
# # Import pre-trained model from Keras
# Import a model pre-trained on ImageNet dataset. The parameter include_top=False
# excludes the final classification layer for ImageNet - which we will replace.


# %%
from tensorflow.keras.applications import InceptionResNetV2

pre_model = InceptionResNetV2(
    weights='imagenet', include_top=False, input_shape=(150, 150, 3))
pre_model.summary()

# %%
