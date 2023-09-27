from flask import Flask, render_template, request

import base64
import re
import tensorflow as tf
import numpy as np
app = Flask(__name__)

model = tf.keras.models.load_model('static/model/mnist_model.h5')

@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/upload", methods = ['POST'])
def upload_file():
    imgstr = re.search(b"base64,(.*)", request.get_data()).group(1)
    img_decode = base64.decodebytes(imgstr)
    with open("output.jpg", "wb") as file:
        file.write(img_decode)

    img_raw = "output.jpg"
    image = tf.io.read_file(img_raw)
    image = tf.image.decode_jpeg(image, channels =1)
    image = tf.image.resize(image, [28,28])
    image = tf.reshape(image, (1,28,28,1))

    with open("fixed.jpg", "wb") as file:
        file.write(image)
    
    probabilities = model.predict(image)
    prediction = np.argmax(probabilities, axis = 1)
    
    return str(prediction)
    