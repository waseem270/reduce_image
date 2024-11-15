from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    confidence = request.form.get('confidence')
    probability = request.form.get('probability')

    if image.filename == '':
        return "No file selected", 400

    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        probability = float(probability)
        
        compressed_image_path = reduce_image(image_path, probability)
        
        return render_template('index.html', download_link=os.path.basename(compressed_image_path))

    return "Upload failed", 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True)

def reduce_image(file_name, np_value):
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)
    pca = PCA(n_components=np_value)
    transformed_image = pca.fit_transform(gray_image)
    reconstructed_image = pca.inverse_transform(transformed_image)
    
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)

    if not os.path.exists(app.config['COMPRESSED_FOLDER']):
        os.makedirs(app.config['COMPRESSED_FOLDER'])

    compressed_image_path = os.path.join(app.config['COMPRESSED_FOLDER'], 'compressed_image.jpg')
    io.imsave(compressed_image_path, compressed_image_uint8)
    return compressed_image_path

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
