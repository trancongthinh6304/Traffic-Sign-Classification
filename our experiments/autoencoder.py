# import libraries
from IPython.display import Image, display
import numpy as np
import os
from os.path import join
from PIL import ImageFile
import pandas as pd
from matplotlib import cm
import seaborn as sns
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
import re
from tqdm import tqdm
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
import tensorflow as tf
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.style.use('fivethirtyeight')
%matplotlib inline
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image
import glob

import warnings;
warnings.filterwarnings('ignore')

#capture paths to images
face_images = glob.glob('lfw/**/*.jpg')

all_images = []

for i in tqdm(face_images):
  img = image.load_img(i, target_size = (75, 75, 3))
  img = image.img_to_array(img)
  img = img/255.0
  all_images.append(img)

all_images = np.array(all_images)

# split data into train and validation data
train_x, val_x = train_test_split(all_images, random_state = 42, test_size = 0.1)

def pixalate_image(image, scale_percent = 40):
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)

  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
  width = int(small_image.shape[1] * 100 / scale_percent)
  height = int(small_image.shape[0] * 100 / scale_percent)
  dim = (width, height)

  low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

  return low_res_image

# get low resolution images for the training set
train_x_px = []

for i in range(train_x.shape[0]):
  temp = pixalate_image(train_x[i,:,:,:])
  train_x_px.append(temp)

train_x_px = np.array(train_x_px)


# get low resolution images for the validation set
val_x_px = []

for i in range(val_x.shape[0]):
  temp = pixalate_image(val_x[i,:,:,:])
  val_x_px.append(temp)

val_x_px = np.array(val_x_px)

# get low resolution images for the training set
train_x_px = []

for i in range(train_x.shape[0]):
  temp = pixalate_image(train_x[i,:,:,:])
  train_x_px.append(temp)

train_x_px = np.array(train_x_px)


# get low resolution images for the validation set
val_x_px = []

for i in range(val_x.shape[0]):
  temp = pixalate_image(val_x[i,:,:,:])
  val_x_px.append(temp)

val_x_px = np.array(val_x_px)

Input_img = Input(shape=(76, 76, 3))  
    
#encoding architecture
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D( (2, 2))(x2)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

# decoding architecture
x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same')(x1)

autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("--------------------Building model--------------------")
autoencoder.summary()
print("--------------------Done--------------------")

early_stopper = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 4, verbose = 1, mode = 'auto')

a_e = autoencoder.fit(train_x_px, train_x,
            epochs = 50,
            batch_size = 256,
            shuffle = True,
            validation_data = (val_x_px, val_x),
            callbacks = [early_stopper])

predictions = autoencoder.predict(val_x_px)

n = 5
plt.figure(figsize = (20, 10))

for i in range(n):
  ax = plt.subplot(2, n, i+1)
  plt.imshow(val_x_px[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax = plt.subplot(2, n, i+1+n)
  plt.imshow(predictions[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Saving model at autoencoder.h5")
autoencoder.save('autoencoder.h5')

root_dir = '../input/road-sign-recognition/AIJ_2gisPUBLISH/AIJ_2gis/train_images/'
sign_images = os.listdir(root_dir)

# img = image.load_img(root_dir+sign_images[0], target_size=(80, 80))
img = image.load_img(root_dir+sign_images[0], target_size=(80, 80, 3))
img = image.img_to_array(img)
img = img/255.0

result = autoencoder.predict(img[None])

plt.rcParams["axes.grid"] = False
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Encoded")
plt.imshow(result[0])

plt.show()