#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kaggle
import zipfile
import numpy as np
import pandas as pd
import os
import cv2
import random
import unittest
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    MaxPooling2D,
    Conv2D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


# In[2]:


get_ipython().system("kaggle competitions download -c digit-recognizer")


# In[3]:


DIR = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\digit-recognizer"


# In[4]:


def data_loader(DIR):
    train_data = pd.read_csv(os.path.join(DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DIR, "test.csv"))
    return train_data, test_data


# In[5]:


train_data, test_data = data_loader(DIR)


# ### Visualize data

# In[6]:


train_data


# In[7]:


x_train, y_train = np.array(train_data.loc[:, "pixel0":]), np.array(train_data.label)


# In[8]:


x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=0)


# In[9]:


test_data


# ### Shape data for training

# In[10]:


x_test = np.array(test_data.loc[:, "pixel0":])


# In[11]:


x_train_shuffled = x_train.reshape(42000, 28, 28)
x_test = x_test.reshape(28000, 28, 28)


# In[12]:


plt.imshow(x_train_shuffled[0], cmap="gray")


# In[13]:


x_train_shuffled_gray = x_train.reshape(42000, 28, 28, 1)
x_test_gray = x_test.reshape(28000, 28, 28, 1)
x_train_shuffled_gray = x_train_shuffled_gray / 255.0
x_test_gray = x_test_gray / 255.0


# In[14]:


y_train_shuffled = tf.keras.utils.to_categorical(y_train, num_classes=10)


# In[15]:


plt.imshow(x_train_shuffled[1], cmap="gray")


# In[16]:


y_train_shuffled[1]


# ### Check shapes

# In[17]:


class TestShapes(unittest.TestCase):
    def test_shapes(self):
        expect_train_data_shape = (42000, 785)
        expect_test_data_shape = (28000, 784)
        expect_x_train_shuffled_shape = (42000, 28, 28)
        expect_y_train_shuffled_shape = (42000,)
        expect_x_train_shuffled_gray_shape = (42000, 28, 28, 1)
        expect_x_test_gray_shape = (28000, 28, 28, 1)
        expect_y_train_shuffled_shape = (
            42000,
            10,
        )  # each label is a size 10 vector one hot encoding

        self.assertEqual(expect_train_data_shape, train_data.shape)
        self.assertEqual(expect_test_data_shape, test_data.shape)
        self.assertEqual(expect_x_train_shuffled_shape, x_train_shuffled.shape)
        self.assertEqual(expect_y_train_shuffled_shape, y_train_shuffled.shape)
        self.assertEqual(
            expect_x_train_shuffled_gray_shape, x_train_shuffled_gray.shape
        )
        self.assertEqual(expect_x_test_gray_shape, x_test_gray.shape)
        self.assertEqual(expect_y_train_shuffled_shape, y_train_shuffled.shape)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)


# ### Create model

# In[18]:


model_1 = Sequential(
    [
        Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ]
)


# In[19]:


model_1.summary()


# In[20]:


model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# ### Train model

# In[21]:


history = model_1.fit(
    x_train_shuffled_gray,
    y_train_shuffled,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
)


# ### Plot loss

# In[22]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])


# ### Generate Predictions

# In[23]:


results = model_1.predict(x_test_gray, verbose=1)
results = results.argmax(axis=1)


# ### Submit output

# In[24]:


submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
submission["Label"] = results
submission.to_csv("predictions.csv", index=False)


# In[ ]:
