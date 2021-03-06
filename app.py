from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

#Load models
modelvgg =load_model('Models/VGG.h5')
modelvgg._make_predict_function() 

modelresnet =load_model('Models/RESNET.h5')
modelresnet._make_predict_function()

# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Go to http://127.0.0.1:5000/')


def model_predict(img_path, model):
    IMGS=256
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMGS, IMGS))
    img = img.reshape(-1, IMGS, IMGS, 3)

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.errorhandler(Exception)
def handle_500(e):
    original = getattr(e, "original_exception", None)
    print(e)
    print("$$$$$$$$$$$")
    app.logger.exception(e)
    return render_template('index.html'), 500

cache={}
@app.route('/modelSelection',methods=['POST', 'GET'])
def select():
    if request.method == 'POST':
        f=request.form['models']
        print(f)
        # MODEL_PATH ='models/CNN.h5'
        model_selected =""
        if f=='RESNET':
            cache['model'] = modelresnet
            model_selected = "Đã chọn Model based on ResNet"
        elif f=='VGG':
            cache['model'] = modelvgg
            model_selected = "Đã chọn Model based on VGG"

        #model= load_model(MODEL_PATH)
        #model._make_predict_function()
        #cache['model'] = model
        #return redirect('/')
        return render_template('index.html', message=model_selected)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']


        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, cache['model'])

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result=""
        if preds[0][0] > 0.5:
        	result = "PNEUMONIA"
        else:
        	result = "NORMAL"
        percent = str(preds[0][0])
        print(result+" ("+percent+")")               # Convert to string
        return result+" ("+percent+")"
    return None


if __name__ == '__main__':
    app.run(debug=True)

