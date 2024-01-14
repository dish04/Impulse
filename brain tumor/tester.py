import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import (CategoricalAccuracy, F1Score, Precision,
                                      Recall)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('MRI_Model')

print("Model loaded")

img = cv2.imread('/Users/dishantharya/new code/neural nets/test2.jpg')
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
resize = tf.image.resize(img_tensor, (256,256))

yhat = model.predict(np.expand_dims(resize/255, 0))

maxi = yhat[0][1]
for i in yhat[0]:
    maxi = max(maxi,i)

for i in range(0,4):
    if(yhat[0][i]==maxi):
        print(i+1)

print(maxi)