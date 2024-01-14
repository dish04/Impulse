import cv2
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Loading Model
model = tf.keras.models.load_model('/Users/dishantharya/new code/MRI_Model')

#Loading Image

data_gen = ImageDataGenerator(rescale=1./255)
path = '/Users/dishantharya/new code/neural nets/sm_test'
test = data_gen.flow_from_directory(directory=path, target_size=(256, 256), batch_size=10, shuffle=False)

def generate_saliency_map(tester, class_index):
    with tf.GradientTape() as tape:
        model.evaluate(tester)
        yhat = model.predict(tester)
        sc = yhat[:, class_index]
    
    dsc_di = tape.gradient(sc, tester)
    return dsc_di

saliency_map = generate_saliency_map(test,2)

saliency_map = tf.maximum(saliency_map,0)

saliency_map = tf.reduce_max(saliency_map, axis=-1)

saliency_map /= tf.reduce_max(saliency_map)

plt.subplot(1,2,1)
plt.imshow(test[:,:,2])
plt.title('Resized Image')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map[0], cmap='turbo')
plt.title('Saliency Map')

plt.show()