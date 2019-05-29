# -*- coding: utf-8 -*-
import base64
import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from process_result_problem_1 import process_result_problem_1
from process_result_problem_2 import process_result_problem_2
from datn_version_2.predict import Cephalometric

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = dir_path + '/test'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/deeplearning")
def deep():
    return render_template("deep.html")


@app.route("/problem-1")
def problem1():
    return render_template("problem-1.html")


@app.route("/result-1", methods=['GET', 'POST'])
def result1():
    if request.method == 'POST':
        fasta_seq = request.form['seq']
        dataset = request.form['options']
        name, sequence, result = process_result_problem_1(fasta_seq, dataset)
        return render_template("result-1.html", name=name, sequence=sequence, result=result)


@app.route("/problem-2")
def problem2():
    return render_template("problem-2.html")


@app.route("/result-2", methods=['GET', 'POST'])
def result2():
    if request.method == 'POST':
        fasta_seq = request.form['seq']
        threshold = float(request.form.get('threshold'))
        name, sequence, y_pred_prob, y_pred_label = process_result_problem_2(fasta_seq, threshold)
        return render_template(
            "result-2.html",
            name=name,
            sequence=sequence,
            sequence_dict=enumerate(list(sequence)),
            sequence_length=len(sequence),
            threshold=threshold,
            y_pred_prob=y_pred_prob,
            y_pred_label=y_pred_label
        )


@app.route("/problem-3")
def problem1():
    return render_template("problem-3.html")


def draw_image(img, coords):
    for coord in coords:
        cv2.circle(img, (int(coord[0]), int(coord[1])), 3, (255, 0, 0), 3)
    return img


@app.route("/result-3", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        print('----------------------')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        img_stream = file.stream
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        coords = predictor.predict(img)
        img = draw_image(img, coords).tobytes()
        jpg_as_text = base64.b64encode(img).decode('utf-8')
        return render_template("result-1.html", image_result="data:image/jpeg;base64," + jpg_as_text)


if __name__ == "__main__":
    predictor = Cephalometric()
    app.run(host='0.0.0.0')
