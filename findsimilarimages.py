#IMPORTANT, test_data folder for dataset should just be images with no subfolders

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
import time
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return img, x


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

#This is where you load user image
user_img, x = load_image("newtest\ISIC_melanoma_dermascopic_0015204.jpg")
new_features = feat_extractor.predict(x)


#goes through dataset images, this is the path of the dataset
folder_path = "test_data"
image_extensions = ['.jpg', '.png']
max_num_images=4000
images= [os.path.join(dp,f) for dp, dn, filenames in os.walk(folder_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images= [images[i] for i in sorted(random.sample(xrange(len(images)),max_num_images))]

#can comment this if you want
print("keeping %d images to analyze" % len(images))

#processes all image and uses the pretained model to analyze the image dataset

#THIS IS WHERE YOU LOAD IMAGE TO PUT INTO PCA TO ANALYZE

tic = time.process_time()
features = []
for i, folder_path in enumerate(images):
    if i % 500 ==0:
        toc = time.process_time()
        elap = toc-tic;
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i,len(images),elap))
        tic = time.process_time()
    img, x = load_image(folder_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

#can also comment this line out if you want
print('finished extracting features for %d images' % len(images))


#performs PCA
features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)
#projecting user image into pca space
new_pca_features = pca.transform(new_features)[0]


#gets the closest distance to target image
def get_closest_images(query_image_idx, num_results=5):
    distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest


def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

#grabs a random indexed image to find similar images
#query_image_idx = int(len(images) * random.random())
 
distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
idx_closest = sorted(range(len(distances)),key=lambda k: distances[k])[0:5]
results_image = get_concatenated_images(idx_closest,200)
#idx_closest=get_closest_images(query_image_idx)
#query_image = get_concatenated_images([query_image_idx], 300)
#results_image = get_concatenated_images(idx_closest, 200)

#displays the results
plt.figure(figsize=(5,5))
plt.title("User input image")
plt.imshow(user_img)
plt.show()

#displays the next 5 images that are similar to target image
plt.figure(figsize = (16,12))
plt.title("result images")
plt.imshow(results_image)
plt.show()


