from math import dist
import os
import keras
from numpy.core.defchararray import asarray
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from scipy.spatial import distance
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 


if False: 
    files = []
    for r, d, f in os.walk("./static/dataset/"):
        for file in f:
            if ('.jpg' in file):
                exact_path = r + file
                files.append(exact_path)

    model = tf.keras.models.load_model('./saved_model/test_model_200_128x128_1.h5')

    representations = []
    for img_path in files:
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.array([img_arr])  # Convert single image to a batch.

        embedding = model.predict(img_arr)[0,:]

        representation = []
        representation.append(img_path)
        representation.append(embedding)
        representations.append(representation)

    embedding_size = 128 #FaceNet output size
    t = AnnoyIndex(embedding_size, 'euclidean')
    
    for i in range(0, len(representations)):
        representation = representations[i]
        img_path = representation[0]
        embedding = representation[1]
        
    t.add_item(i, embedding)
    
    t.build(3) #3 trees

    #save the built model
    t.save('result.ann')
 
#restore the built model
embedding_size = 3
t = AnnoyIndex(embedding_size, 'euclidean')
t.load('result_annoy.ann')

idx = 0 #0 index item is the target
k = 3 #find the k nearest neighbors
neighbors = t.get_nns_by_item(idx, k+1)

files = []
for r, d, f in os.walk("./static/dataset/"):
    for file in f:
        if ('.jpg' in file):
            exact_path = r + file
            files.append(exact_path)

fig=plt.figure(figsize=(10,10))
rows = 2
columns = 2

for i in range(0,len(neighbors)): 
    fig.add_subplot(rows, columns, i+1)
    image=mpimg.imread(files[neighbors[i]])
# showing image
    plt.imshow(image)
    plt.axis('off')
    plt.title("First")
    plt.imshow(image)
plt.show()
print(neighbors)
