from flask import(
	Flask,
	render_template,
	redirect,
	url_for,
	session,
	request,
	flash
	)
import helperfunctions


app = Flask(__name__)

app.secret_key = str(helperfunctions.sessionID())

@app.route("/")
def direct():
	return redirect(url_for('home'))

@app.route('/home')
def home():
	stringsList = []
	return render_template('home.html', list = stringsList)
	

