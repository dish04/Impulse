import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import (CategoricalAccuracy, F1Score, Precision,
                                      Recall)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('/Users/dishantharya/new code/MRI_Model')

print("Model loaded")

data_test = '/Users/dishantharya/new code/neural nets/brain tumor/MRI/Testing'
data_gen = ImageDataGenerator(rescale=1./255)
test = data_gen.flow_from_directory(directory=data_test, target_size=(256, 256), batch_size=10, shuffle=False)

#print(tf.shape(np.expand_dims(resize/255,0)))

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
f1 = F1Score()

test_dataset = tf.data.Dataset.from_generator(
    lambda: test,
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)

for batch_num, batch in enumerate(test):
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    f1.update_state(y, yhat)
    print(f"Processed batch {batch_num}/{len(test)}")
    if batch_num+1 >132:
        break

print(f"Precision: {pre.result().numpy()}")
print(f"Recall:{re.result().numpy()}")
print(f"Categorical Accuracy:{acc.result().numpy()}")
print(f"F1 Score:{f1.result().numpy()}")