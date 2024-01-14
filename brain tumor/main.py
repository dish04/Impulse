import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
from PIL import UnidentifiedImageError
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_train = '/Users/dishantharya/new code/neural nets/brain tumor/MRI/Training'
data_test = '/Users/dishantharya/new code/neural nets/brain tumor/MRI/Testing'

def load_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                image_path = os.path.join(root, file)
            except UnidentifiedImageError:
                print(f"Skipping corrupted file: {image_path}")
                os.remove(image_path)

#load_images(data_train)
#load_images(data_train)

data_gen = ImageDataGenerator(rescale=1./255)

train = data_gen.flow_from_directory(directory=data_train, target_size=(256, 256), batch_size=10)
test = data_gen.flow_from_directory(directory=data_test, target_size=(256, 256), batch_size=10, shuffle=False)

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
    )

history = model.fit(train , epochs=10, validation_data = test)

predictions = model.predict(test)

model.save('MRI_Model')