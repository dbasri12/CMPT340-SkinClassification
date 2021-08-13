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

app = Flask(__name__)

@app.route("/")
def direct():
	return redirect(url_for('home'))

@app.route('/home')
def home():
	stringsList = []
	class_names = ['1','2','3','4','5','6']
	img_height = 180
	img_width = 180
	model = tf.keras.models.load_model('./saved_model/my_model.h5')
	model.summary()
	data_dir = pathlib.Path("skin_photo")
	img_path = tf.keras.utils.get_file("./test_data/ISIC_melanoma_dermascopic_0015219.jpg")
	img = keras.preprocessin.image.load_img(img_path, target_size=(img_height, img_width))
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	print("{} {:.2f}".format(class_names[np.argmax], 100*np.max(score)))
	return render_template('home.html', list = stringsList)
	

