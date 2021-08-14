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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
from werkzeug.utils import secure_filename

LINK = os.path.join('static', 'user_input')

app = Flask(__name__)

UPLOAD_FOLDER = './static/user_input'
 
app.config['USER_INPUT'] = LINK
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

		#Analyze Input Photo
		path_name = "./static/user_input/"+filename
		img = tf.keras.preprocessing.image.load_img(path_name, target_size=(img_height, img_width))
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0)
		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		prediction = str(class_names[np.argmax(score)])
		predictionPercent = str(round(100*np.max(score)))
		predictionMessage = "The skin condition is diagnosed as "+prediction+" with a "+predictionPercent+" percent confidence."
		print("{} {:.2f}".format(class_names[np.argmax(score)], 100*np.max(score)))

		#Render Template
		full_filename = os.path.join(app.config['USER_INPUT'], filename)
		print(full_filename)
		return render_template('homeNew.html', img = full_filename, prediction = predictionMessage)
	return render_template('homeNew.html')
	

