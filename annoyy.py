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
from tensorflow.keras.applications import vgg16
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
from deepface.commons import functions
from deepface.basemodels import Facenet


files = []
for r, d, f in os.walk("./static/dataset/"):
    for file in f:
        if ('.jpg' in file):
            exact_path = r + file
            files.append(exact_path)

if False:
    

    #model = tf.keras.models.load_model('./saved_model/test_model_200_128x128_1.h5')
    model = tf.keras.models.load_model('./saved_model/my_model.h5')
    #vgg_model= vgg16.VGG16(weights='imagenet')
    #facenet_model=Facenet.loadModel()

    representations = []
    for img_path in files:
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(180, 180))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.array([img_arr])  
        #img_batch = np.expand_dims(img_arr, axis=0)
        # Convert single image to a batch.

        #processed_img=vgg16.preprocess_input(img_batch.copy())

        #embedding = vgg_model.predict(processed_img)[0,:]
        
        #img = functions.preprocess_face(img=img_path, target_size=(160, 160))
        embedding = model.predict(img_arr)[0,:]
        #embedding = vgg_model.predict(processed_img)[0,:]

        representation = []
        representation.append(img_path)
        representation.append(embedding)
        representations.append(representation)
    embedding_size = 3 #FaceNet output size
    t = AnnoyIndex(embedding_size, 'euclidean')
    
    for i in range(0, len(representations)):
        representation = representations[i]
        img_path = representation[0]
        embedding = representation[1]
        
        t.add_item(i, embedding)
    
    t.build(3) #3 trees
    t.save('result_annoy_vgg.ann')

    #save the built model


 
#restore the built model
embedding_size = 3
t = AnnoyIndex(embedding_size, 'euclidean')
t.load('result_annoy_vgg.ann')


idx = 17 #0 index item is the target
k = 3 #find the k nearest neighbors
neighbors = t.get_nns_by_item(idx, k+1)


fig=plt.figure(figsize=(10,10))
rows = 2
columns = 2

for i in range(0,len(neighbors)): 
    fig.add_subplot(rows, columns, i+1)
    image=mpimg.imread(files[neighbors[i]])
# showing image
    plt.imshow(image)
    plt.axis('off')
    if i==0:
        plt.title("Query image: ")
    if(files[neighbors[i]].find("melanoma")!= -1):
        plt.title("Melanoma")
    if(files[neighbors[i]].find("nevus")!= -1):
        plt.title("Nevus")
    if(files[neighbors[i]].find("seborrheic")!= -1):
        plt.title("Seborrheic Kertosis")
    plt.imshow(image)
plt.show()
print(neighbors)
