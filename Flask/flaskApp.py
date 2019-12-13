# This file will be used as the backend to serve our model

# Resources:
# https://palletsprojects.com/p/flask/
#

# render_template lets us define a html file and call it
# request is used for handling the get/post/set requests
from flask import Flask, render_template, request

# We're going the take the image that the user draws and
# reshape it using the methods: imsave, imread, imresize
from scipy.misc import imsave, imread, imresize

# Imports
import numpy as np
import sys
import os
import base64
import re
import tensorflow as tf
from tensorflow.keras.models import load_model

#from tensorflow.keras.models import Sequential, load_model

# This will initialise Flask
app = Flask(__name__)


# Declared these two global variables
global model, graph


# This will load the model
model = load_model('model.h5')

# This will get the graph
graph = tf.get_default_graph()


# This will decode the image from base64 into raw binary data
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('result.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


# This is how we tell the application what happens if the user
# goes to a certain address.
@app.route('/')
def index():
    return render_template("index.html")


# This will be used when the user clicks submit after they draw a digit
@app.route('/predict', methods=['GET', 'POST'])
def predit():
    # This will get the raw serialized data from the image
    imgData = request.get_data()
    # This will encode it into a usable format
    convertImage(imgData)
    # This reads it into memory
    x = imread('result.png', mode='L')
    # Changes the size of image
    x = imresize((x, (28, 28))/255)
    # This will save the new resized image
    imsave('drawing.jpg', x)
    # This changes the shape into a 4D tensor - this is fed into the model
    x = x.reshape(1, 28, 28, 1)
    # This is the computation graph
    with graph.as_default():
        # Performs the prediction
        result = model.predict(x)
        # Converts the response to a string
        response = np.argmax(out, axis=1)
        # Returns the response as a string
        return str(response[0])


# This will run the application on port 5000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=False)
