# import the Flask class from the flask module
from flask import Flask, render_template, send_file, request, session, redirect, url_for
import numpy as np
import pandas as pd
from copy import deepcopy
from werkzeug import secure_filename
from PIL import Image
from io import StringIO
from keras.models import Model, load_model
from keras import backend as K
from predict import predict
import os


ALLOWED = set(['tiff'])

# create the application object
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=["GET","POST"])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed(file.filename):
            f = request.files['file']
            full_input_path = 'static/' + str(secure_filename(f.filename))
            f.save(full_input_path)
            bw = Image.open(full_input_path).convert('LA')
            bwpath = 'static/bw.tiff'
            bw.save(bwpath)
            model = load_model('ecDNA_model_dilated_context.h5')
            pred = predict(model, full_input_path)
            K.clear_session()
            session['predpath'] = pred
            session['bwpath'] = bwpath
            session['origpath'] = full_input_path
            return redirect(url_for('result'))
    return render_template('upload.html')  # render a template

@app.route('/result')
def result():
    try:
        pred = session['predpath']
        bwpath = session['bwpath']
        origpath = session['origpath']
        print(pred, bwpath, origpath)
        return render_template('result.html', pred=pred, imgpath=origpath, bw=bwpath, error="")
    except:
        return render_template('result.html', pred="", imgpath="", bw="", error="**No upload received.**")

@app.route('/download')
def download():
    array = np.load('pred_3.tiff.npy')
    np.savetxt('dataset.csv', array.astype(np.int), fmt='%d', delimiter=',')
    return send_file('dataset.csv', mimetype='text/csv',attachment_filename='pred.csv', as_attachment=True)

@app.route('/carousel')
def carousel():
    return render_template('carousel.html', input1='static/3.tiff', input2='static/bw.tiff', input3='static/pred_3.tiff')

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)

