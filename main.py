# -*- coding: utf-8 -*-
import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from process_result_problem_1 import process_result_problem_1
from process_result_problem_2 import process_result_problem_2
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

@app.route("/result-1", methods = ['GET', 'POST'])
def result1():
	if request.method == 'POST':
		fasta_seq = request.form['seq']
		dataset = request.form['options']
		name, sequence, result = process_result_problem_1(fasta_seq, dataset)
		return render_template("result-1.html",name = name, sequence = sequence, result = result)

@app.route("/problem-2")
def problem2():
    return render_template("problem-2.html") 

@app.route("/result-2", methods = ['GET', 'POST'])
def result2():
	if request.method == 'POST':
		fasta_seq = request.form['seq']
		threshold = float(request.form.get('threshold'))
		name, sequence, y_pred_prob, y_pred_label = process_result_problem_2(fasta_seq, threshold)
		return render_template(
			"result-2.html",
			name = name,
			sequence = sequence,
			sequence_dict = enumerate(list(sequence)),
			sequence_length = len(sequence),
			threshold = threshold,
			y_pred_prob = y_pred_prob,
			y_pred_label = y_pred_label
		)

if __name__ == "__main__":
	app.run(debug = True)
