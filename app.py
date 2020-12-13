from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

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

# Model saved with Keras model.save()
#MODEL_PATH = './model.h5'
modelvgg =load_model('Models/VGG.h5')
modelvgg._make_predict_function() 
# Load your trained model
modelresnet =load_model('Models/RESNET.h5')
modelresnet._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Go to http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
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
            model_selected = "Đã chọn ResNet"
        elif f=='VGG':
            cache['model'] = modelvgg
            model_selected = "Đã chọn VGG16"

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
        if preds[0][0] > 0.85:
        	result = "PNEUMONIA"
        else:
        	result = "NORMAL"
        percent = str(preds[0][0])
        print(result+" ("+percent+")")               # Convert to string
        return result+" ("+percent+")"
    return None


if __name__ == '__main__':
    app.run(debug=True)

