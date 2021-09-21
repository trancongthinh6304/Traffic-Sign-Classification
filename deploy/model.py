import pandas as pd
import numpy as np
import model
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
# from tensorflow.keras.applications import ResNet50
from PIL import Image
import json
import tensorflow.keras.backend as K
import cv2
import os
from keras.layers import Input, Lambda
def preprocess_image(url):
    img = image.load_img(url, target_size=(80,80,3))
    img = image.img_to_array(img)
    img =np.expand_dims(img/255.,axis=0)
    return img

def load_index_to_label_dict(path='index_to_class_label.json'):
    """Retrieves and formats the index to class label lookup dictionary needed to 
    make sense of the predictions. When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict

def predict(image,model):
    predictions = model.predict(image)    
    return predictions

def load_model(path):
    model = tf.keras.models.load_model(path, compile=False)
    # model.make_predict_function()
    return model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def top(scores,class_names, n=5):
    top_scores=[]
    top_scores_percentage=[]
    for i in range(n):
        top_scores.append(class_names[np.argmax(scores)])
        top_scores_percentage.append(str(100*np.max(scores)))
        scores = np.delete(scores, np.argmax(scores))
        class_names = np.delete(class_names,np.argmax(scores))

    return top_scores, top_scores_percentage

def rescale_img(img): #return a [0-255] image
  new_arr = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
  img_new = np.zeros(shape=(80,80,3), dtype= np.int16)
  img_new[..., 0] = new_arr[...,2]
  img_new[...,1]=new_arr[...,1]
  img_new[..., 2] = new_arr[...,0]
  return img_new
