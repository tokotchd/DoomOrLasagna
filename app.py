import os
from flask import Flask, render_template, request
from ModelRunner import ModelRunner
import numpy as np
from werkzeug.utils import secure_filename
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/temp/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['host'] = '0.0.0.0'
model_runner = ModelRunner('./export')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/what', methods=['GET'])
def what():
    return render_template('what.html')

@app.route('/why', methods=['GET'])
def why():
    return render_template('why.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/doom', methods=['GET'])
def doom():
    return render_template('doom.html')

@app.route('/lasagna', methods=['GET'])
def lasagna():
    return render_template('lasagna.html')

@app.route('/', methods=['POST'])  # classify uses root, but POST rather than GET
def classify_image():
    image_file = request.files.get('file', None)
    if image_file and image_file.filename != '' and allowed_file(image_file.filename):
        local_image_path = app.config['UPLOAD_FOLDER'] + str(secure_filename(image_file.filename))
        image_file.save(local_image_path)
        loaded_image = cv2.imread(local_image_path)
        # now that the data is in memory, remove the file so we don't fill the drive with files.
        os.remove(local_image_path)
        class_output = model_runner.run_on_image(loaded_image)
        class_guess = np.argmax(class_output)
        class_confidence = class_output[class_guess]
        if class_guess == 0:
            class_name = 'lasagna'
        else:
            class_name = 'doom'
        if class_confidence > 0.8:
            class_message = "We're very confident that your submitted image was of: "
        elif class_confidence > 0.65:
            class_message = "We're fairly sure that your submitted image was of: "
        else:
            class_message = "We're not sure, but your submitted image might be of: "
        output_message = class_message + class_name

    else:
        output_message = "Something went wrong with the file uploaded, please try another."
    return render_template('index.html', message=output_message)

    # (failed) attempts at doing in-memory loading
    # image_string = base64.b64encode(image_file.read())
    # image_bytes = base64.decodebytes(image_string)
    # numpy_image = np.frombuffer(image_bytes, dtype=np.uint8)
    # print(numpy_image)

if __name__ == "__main__":
    app.run()