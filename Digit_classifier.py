#!/usr/bin/env python
# coding: utf-8




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





get_ipython().system("kaggle competitions download -c digit-recognizer")





DIR = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\digit-recognizer"





def data_loader(DIR):
    train_data = pd.read_csv(os.path.join(DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DIR, "test.csv"))
    return train_data, test_data





train_data, test_data = data_loader(DIR)







train_data





x_train, y_train = np.array(train_data.loc[:, "pixel0":]), np.array(train_data.label)





x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=0)





test_data


# ### Shape data for training




x_test = np.array(test_data.loc[:, "pixel0":])





x_train_shuffled = x_train.reshape(42000, 28, 28)
x_test = x_test.reshape(28000, 28, 28)





plt.imshow(x_train_shuffled[0], cmap="gray")





x_train_shuffled_gray = x_train.reshape(42000, 28, 28, 1)
x_test_gray = x_test.reshape(28000, 28, 28, 1)
x_train_shuffled_gray = x_train_shuffled_gray / 255.0
x_test_gray = x_test_gray / 255.0





y_train_shuffled = tf.keras.utils.to_categorical(y_train, num_classes=10)





plt.imshow(x_train_shuffled[1], cmap="gray")





y_train_shuffled[1]







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
        )  

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





model_1.summary()





model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])







history = model_1.fit(
    x_train_shuffled_gray,
    y_train_shuffled,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
)







plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])







results = model_1.predict(x_test_gray, verbose=1)
results = results.argmax(axis=1)







submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
submission["Label"] = results
submission.to_csv("predictions.csv", index=False)



