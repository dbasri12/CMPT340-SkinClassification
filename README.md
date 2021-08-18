# Cmpt 340 Project
This project is a website that classifies different skin diseases such as melanoma, seboorheic keratosis and nevus.

## Installation

1\. Ensure pip is installed. If it is not installed, run
```
$ python get-pip.py
```
2\. Install necessary libraries
```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
3\. To deploy the website locally, run
```
$ cd <path to project>
$ flask run
```
The website should now be running on local host port 5000 by default: http://localhost:5000

If this does not work, try these steps.

1\. Activate the virtual enviroment named env using powershell (env\Scripts\activate)
```
$ python -m venv env
$ env\Scripts\activate
```
2\. Compile code using powershell (python app.py) 
```
$ python app.py
```
3\. Set app as flask ($env:FLASK_APP="app.py")
```
$ $env:FLASK_APP="app.py"
```
4\. Run Flask
```
$ flask run
```

## Usage
1\. Click Choose File button and open an image of a skin disease <br />
2\. Click submit <br />
3\. Wait for results

## Notes
* add images to the folder static/dataset to be used for image similarity
* to retrain the model, put images of each category into its own folder under the skin_photo folder for example:<br />
skin_photo/ <br />
&emsp; nevus/ <br />
&emsp; melanoma/ <br />
&emsp; keratosis/ <br />
