import numpy as np
from PIL import Image, ImageOps
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import tensorflow as tf
import scipy.misc
from keras.datasets import cifar10
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageEnhance
except ImportError:
    import Image


def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# load the CIFAR10 data
(x_train, _), (x_test, _) = cifar10.load_data()

directory = './dataset/test/0/'
x_test = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode=None,
    class_names='binary',
    color_mode="rgb",
    batch_size=500,
    image_size=(256, 256),
    shuffle=True,
    seed=256,
    validation_split=0.01,
    subset='validation',
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

x_test = np.stack(list(x_test))
x_test = x_test[0]

# input image dimensions
# we assume data format "channels_last"
img_rows = x_test.shape[1]
img_cols = x_test.shape[2]
channels = x_test.shape[3]

# create saved_images folder
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
"""
# display the 1st 100 input images (color and gray)
imgs = x_test[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs.astype('uint8'), interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

# convert color train and test images to gray
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# display grayscale version of test images
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()
"""
i = 0
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(x_test[i].astype("uint8"))
gray = cv2.cvtColor(x_test[i].astype("uint8"), cv2.COLOR_RGB2BGR)
cv2.imwrite('Images/origin.png', gray)
print(x_test[i].astype("uint8"))
print(".......................................................................")
print(x_test[i])
plt.show()

x_test = rgb2gray(x_test)
x_test_gray = x_test.astype('float32') / 255

plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(x_test_gray[i])
gray = cv2.cvtColor(x_test[i].astype("uint8"), cv2.COLOR_RGB2BGR)
cv2.imwrite('Images/gray.png', gray)
print(x_test_gray[i].astype("uint8"))
print(".......................................................................")
print(x_test_gray[i])
plt.show()

# reshape images to row x col x channel for CNN input

x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

#import model created from the training session
autoencoder = load_model('./saved_model/colorized_ae_model.027.h5')

#make a prediction and display the results
x_decoded = autoencoder.predict(x_test_gray)
x_decoded = np.multiply(x_decoded, 255)
gray = cv2.cvtColor(x_decoded[i].astype("uint8"), cv2.COLOR_RGB2BGR)
cv2.imwrite('Images/color.png', gray)
print(type(x_decoded[i]))
print(x_decoded[i].shape)
print(".......................................................................")
print(x_decoded[i].astype('uint8'))
plt.imshow(x_decoded[i].astype('uint8'))
plt.axis("off")
plt.show()

#post-process model output images
background = Image.open("Images/gray.png")
overlay = Image.open("Images/color.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)

enhancer = ImageEnhance.Brightness(new_img)
factor = 1.15
im_output = enhancer.enhance(factor)

enhancer = ImageEnhance.Sharpness(im_output)

factor = 3
im_s_1 = enhancer.enhance(factor)

enhancer = ImageEnhance.Color(im_s_1)

factor = 1.5
im_s_1 = enhancer.enhance(factor)

enhancer = ImageEnhance.Contrast(im_s_1)

factor = 1.5
im_s_1 = enhancer.enhance(factor)

im_s_1.save('Images/final.png');