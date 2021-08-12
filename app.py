from flask import(
	Flask,
	render_template,
	redirect,
	url_for,
	session,
	request,
	flash
	)


app = Flask(__name__)

@app.route("/")
def direct():
	return redirect(url_for('home'))

@app.route('/home')
def home():
	stringsList = []
	return render_template('home.html', list = stringsList)
	

