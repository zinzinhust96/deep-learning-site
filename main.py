# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from process_result_problem_1 import process_result_problem_1
from process_result_problem_2 import process_result_problem_2
from ast import literal_eval
from ensemble import load_trained_models_2, ensemble_folds
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

# load in problem 2 models
FEATURES_2_SHAPE = (21, 20, 1)
MODEL_PATH_2 = os.path.dirname(os.path.realpath(__file__)) + '/problem-2/seed_19'
trained_models_2 = load_trained_models_2(input_shape = FEATURES_2_SHAPE, directory = MODEL_PATH_2)
ensemble_model_2 = ensemble_folds(trained_models_2, input_shape = FEATURES_2_SHAPE)
graph = tf.get_default_graph()

@app.route("/")
def index():
	return render_template("index.html")

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
		threshold = float(request.form['options'])
		email = request.form['email']
		global graph
		with graph.as_default():
			name, sequence, y_pred_prob, y_pred_label = process_result_problem_2(fasta_seq, threshold, ensemble_model_2)
			if email:
				message = Mail(
					from_email='quangnguyenhong@admin.hust.edu.vn',
					to_emails=email,
					subject='Predicting Protein-DNA Binding Residues - Result',
					html_content=render_template(
						"result-2-mail.html",
						name = name,
						sequence = sequence,
						sequence_dict = enumerate(list(sequence)),
						sequence_length = len(sequence),
						threshold = threshold,
						y_pred_prob = y_pred_prob,
						y_pred_label = y_pred_label
					)
				)
				sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
				response = sg.send(message)
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

@app.route("/report-2", methods = ['GET', 'POST'])
def report2():
	if request.method == 'POST':
		name = request.form['name']
		sequence = request.form['sequence']
		y_pred_prob = request.form['y_pred_prob']
		threshold = float(request.form.get('threshold'))
		y_pred_prob = literal_eval(y_pred_prob)
		y_pred_label = np.where(np.array(y_pred_prob) < threshold, 0, 1)
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
	app.run(host='0.0.0.0', port=5000)
