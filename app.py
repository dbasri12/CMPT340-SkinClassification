from flask import(
	Flask,
	render_template,
	redirect,
	url_for,
	session,
	request,
	flash
	)
import numpy as np
import tensorflow as tf
import keras
from numpy.core.defchararray import asarray
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from scipy.spatial import distance
from sklearn.decomposition import PCA
import random
import os
from werkzeug.utils import secure_filename
import time
import matplotlib.pyplot as plt
from math import dist
import io
from PIL import Image
from flask import Response

app = Flask(__name__)

#folder where user input stored
UPLOAD_FOLDER = './static/user_input'
 
app.config['USER_INPUT'] = os.path.join('static', 'user_input')
app.config['DATASET'] = os.path.join('static', 'dataset')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 9999
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

@app.route("/")
def direct():
	return redirect(url_for('home'))

@app.route('/home', methods = ['GET', 'POST'])
def home():

	#Initialize values, load model
	class_names = ['Seborrheic Kertosis','Nevus','Melanoma']
	img_height = 180
	img_width = 180
	model = tf.keras.models.load_model('./saved_model/my_model.h5')

	#if POST get image input
	if request.method == 'POST':

		#Handle input
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		
		#Save input image
		filename = secure_filename(file.filename)
		filePath = os.path.join(app.config['USER_INPUT'], filename)
		file.save(filePath)

		#Diagnose Input Photo
		path_name = "./static/user_input/"+filename
		img = tf.keras.preprocessing.image.load_img(path_name, target_size=(img_height, img_width))
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0)
		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		prediction = str(class_names[np.argmax(score)])
		predictionPercent = str(round(100*np.max(score)))
		predictionMessage = "The skin condition is diagnosed as "+prediction+" with a "+predictionPercent+" percent confidence."
		
		model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
		def load_image(path):
			img = image.load_img(path, target_size=model.input_shape[1:3])
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			return img, x

		feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

		#Similar photos
		#This is where you load user image
		user_img, x = load_image(path_name)
		new_features = feat_extractor.predict(x)

		#goes through dataset images, this is the path of the dataset
		folder_path = "static\dataset"
		image_extensions = ['.jpg', '.png']
		max_num_images=4000
		images= [os.path.join(dp,f) for dp, dn, filenames in os.walk(folder_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
		if max_num_images < len(images):
			images= [images[i] for i in sorted(random.sample(xrange(len(images)),max_num_images))]

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
 
		distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
		idx_closest = sorted(range(len(distances)),key=lambda k: distances[k])[0:5]
		results_image = get_concatenated_images(idx_closest,200)

		#displays the results
		#plt.figure(figsize=(5,5))
		#plt.title("User input image")
		#plt.imshow(user_img)
		#plt.show()

		#displays the next 5 images that are similar to target image
		#plt.figure(figsize = (16,12))
		#plt.title("result images")
		#plt.imshow(results_image)
		#plt.show()

		#Render Template
		full_filename = os.path.join(app.config['USER_INPUT'], filename)
		close1 = str(images[idx_closest[0]])
		close2 = str(images[idx_closest[1]])
		close3 = str(images[idx_closest[2]])
		print(close1)
		print(close2)
		print(close3)
		listOfDiscriptors = ["N/A","N/A","N/A"]
		if close1.find("nevus") != -1:
			listOfDiscriptors[0] = "Nevus"
		if close1.find("melanoma") != -1:
			listOfDiscriptors[0] = "Melanoma"
		if close1.find("seborrheic") != -1:
			listOfDiscriptors[0] = "Seborrheic Kertosis"

		if close2.find("nevus") != -1:
			listOfDiscriptors[1] = "Nevus"
		if close2.find("melanoma") != -1:
			listOfDiscriptors[1] = "Melanoma"
		if close2.find("seborrheic") != -1:
			listOfDiscriptors[1] = "Seborrheic Kertosis"

		if close3.find("nevus") != -1:
			listOfDiscriptors[2] = "Nevus"
		if close3.find("melanoma") != -1:
			listOfDiscriptors[2] = "Melanoma"
		if close3.find("seborrheic") != -1:
			listOfDiscriptors[2] = "Seborrheic Kertosis"
		return render_template('homeNew.html', img = full_filename, 
											prediction = predictionMessage, 
											result1 = close1, 
											result2 = close2, 
											result3 = close3,
											text1 = listOfDiscriptors[0],
											text2 = listOfDiscriptors[1],
											text3 = listOfDiscriptors[2])
	return render_template('homeNew.html')
	

