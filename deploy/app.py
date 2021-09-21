from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import model
from tensorflow.keras.preprocessing.image import save_img
import numpy as np
app = Flask(__name__)


VGG19 = model.load_model('static/model/VGG19.h5')
autoencoder = model.load_model('static/model/autoencoder.h5')
index_to_class_label_dict = model.load_index_to_label_dict('static/index_to_class_label.json')
class_names = list(index_to_class_label_dict.values())
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        image = model.preprocess_image(path)
        preprocess_autoencoder = model.predict(image,autoencoder)
        rescale_autoencoder = model.rescale_img(preprocess_autoencoder)
        result_VGG19 =model.predict(np.expand_dims(rescale_autoencoder, axis=0)*1/255.0,VGG19) 
        top_scores, top_scores_percentage = model.top(result_VGG19,class_names, n=1)
        save_img(path, preprocess_autoencoder[0])    
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename,top_scores=top_scores,top_scores_percentage=top_scores_percentage)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()
