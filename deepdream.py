import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

img_temp = cv2.imread('/content/sample_data/images.jpg')/255
w, h, _ = img_temp.shape

names = ['mixed3','mixed5']
layers = [model.get_layer(layer_name).output for layer_name in names]
layers_grad = tf.keras.backend.gradients(layers, model.input)

sess = tf.keras.backend.get_session()

octave_value = 1.3  
for i in range(3):
    new_shape = (int(h* octave_value**i), int(w* octave_value**i))
    img_temp = cv2.resize(img_temp, new_shape)
    for _ in range(500):
        grads = sess.run(layers_grad, feed_dict = { model.input : [img_temp]})
        grads /= np.std(grads)
        img_temp = img_temp + grads[0,0,:,:,:]*0.001
        np.clip(img_temp, 0, 255)

plt.imshow(img_temp)
