import cv2
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

#Loading Model
model = tf.keras.models.load_model('/Users/dishantharya/new code/neural nets/brain tumor/MRI_Model')

#Loading Image
input_image = cv2.imread('/Users/dishantharya/new code/neural nets/test31.jpg')
img_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
resize = tf.image.resize(img_tensor, (256,256))

input_image = tf.expand_dims(resize/255, axis=0)

#Creating feature maps
last_conv_layer = model.get_layer('conv2d_2')
feature_map_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

#Derivative
def generate_grad_cam(img_array, class_index):
    #class index is the index of each of the class
    # 0 for glioma
    # 1 for meningioma
    # 2 for notumor
    # 3 for pituitary
    with tf.GradientTape() as tape:
        Ak, predictions = feature_map_model(img_array)
        yc = predictions[:, class_index]
    
    dyc_dAk = tape.gradient(yc, Ak)
    
    #ac
    ac = tf.reduce_mean(dyc_dAk, axis=(1, 2))
    ac = tf.expand_dims(tf.expand_dims(ac,axis=1),axis=1)
    print("Ac SHAPE = ",ac.shape)
    
    #Multiplying feature map and pooled gradients
    heatmap = tf.multiply(Ak, ac)
    print("HEATMAP SHAPE = ",heatmap.shape)
    heatmap = tf.squeeze(heatmap)

    #ReLU activation
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = tf.image.resize(heatmap, (256, 256))
    heatmap = tf.reduce_sum(heatmap, axis=-1, keepdims=True)

    heatmap = heatmap.numpy()
    return heatmap

###GradCAM###
#CLASS_INDEX can be changed here
heatmap = generate_grad_cam(
    img_array=input_image,
    class_index=2
    )

def plot(heatmap,resize,i,r):
    if i == 0:
        name = 'glioma'
    elif i == 1:
        name = 'meningioma'
    elif i == 2:
        name = 'notumor'
    else:
        name = 'pituitary'

    plt.subplot(r,3,1)
    plt.imshow(resize[:,:,2])
    plt.title(f'{name} Image')

    plt.subplot(r,3,2)
    plt.imshow(heatmap[:, :,0], cmap='turbo')
    plt.title(f'{name} GradMap')

    plt.subplot(r,3,3)
    plt.imshow(resize[:,:,2])
    plt.imshow(heatmap[:, :,0], alpha=0.8)
    plt.title(f'{name}')

#Could not plot all the images in one plot, can change path and class index accordingly
plot(heatmap,resize,2,1)

plt.show()