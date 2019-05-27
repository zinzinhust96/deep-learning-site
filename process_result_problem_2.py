import numpy as np
import os
seed = 13
np.random.seed(seed) # for reproducibility
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical, plot_model
from keras.models import Model
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from process_result import calculate_probability
from preprocess2 import WindowSlidePSSMExtractor
import subprocess
import re

K.set_image_data_format('channels_last')

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # # Embeddings layer
    # embeds = layers.Embedding(input_dim = len(STANDARD_AMINO_ACID), output_dim = WINDOW_SIZE, input_length = WINDOW_SIZE)(x)
    # # Reshape tensor
    # new_shape = (K.int_shape(embeds)[1], K.int_shape(embeds)[2], 1) 
    # embeds = layers.Reshape(new_shape)(embeds)

    # embeds = layers.ZeroPadding2D(padding=(1, 1), data_format=None)(embeds)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv1')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.7)(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=7, kernel_initializer='he_normal', strides=2, padding='valid', dropout=0.2)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps', kernel_initializer='he_normal', dropout=0.1)(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    model = models.Model(x, out_caps)

    return model

def load_trained_models(input_shape, directory):
    n_folds = 10
    models = list()
    for i in range(n_folds):
        # define model
        model = CapsNet(input_shape=input_shape, n_class=2, routings=3)
        weight_file = directory + '/fold_%d' % (i) + '/best_model.h5'
        model.load_weights(weight_file)
        print('load weight from ', weight_file)
        models.append(model)

    return models

def ensemble(models, input_shape):
    model_input = layers.Input(shape=input_shape)
    outputs = [model(model_input) for model in models]
    y = layers.Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model

def process_result_problem_2(fasta_seq, threshold):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # read pssm
    name, sequence = fasta_seq.split('\n')
    pssm_matrix = read_pssm(fasta_seq)
    features = WindowSlidePSSMExtractor(pssm_matrix).features
    features = np.array(features, dtype="float32")
    # add one shape to sequence list for standardizing 
    new_shape = list(features.shape)
    new_shape.append(1)
    features = np.reshape(features, new_shape)
    
    trained_models = load_trained_models(input_shape = features.shape[1:], directory = dir_path + '/problem-2/seed_7')
    ensemble_model = ensemble(trained_models, input_shape=features.shape[1:])
    y_pred = ensemble_model.predict(features)
    y_pred_prob = calculate_probability(y_pred)
    y_pred_label = np.where(np.array(y_pred_prob) < threshold, 0, 1)
    K.clear_session()
    return name, sequence, y_pred_prob, y_pred_label

def read_pssm(fasta_seq):
    temp_path = os.path.dirname(os.path.realpath(__file__)) + '/temp/'
    fasta_file = "protein.fasta"
    txt_file = "protein.txt"
    csv_file = "protein.csv"
    print('fasta_seq: ', fasta_seq)
    
    # write to fasta file
    with open(temp_path + fasta_file, 'w') as output_file:
        output_file.write(fasta_seq + '\n')
    
    
    # run psiblast on fasta file to get pssm txt file
    # ./ncbi-blast-2.9.0+/bin/psiblast -db ncbi-blast-2.9.0+/bin/db/swissprot -evalue 0.01 -query $f -out_ascii_pssm ./pssm/$name -num_iterations 3 -num_threads 6
    subprocess.call(["./ncbi-blast-2.9.0+/bin/psiblast", "-db", "ncbi-blast-2.9.0+/bin/db/swissprot", "-evalue", "0.01", "-query", "./temp/" + fasta_file, "-out_ascii_pssm", "./temp/" + txt_file, "-num_iterations", "3"])

    # parse pssm from pssm txt file to csv file\
    parse_pssm(temp_path, txt_file, csv_file)

    # read csv file and return pssm matrix
    pssm_matrix = np.genfromtxt(temp_path + csv_file, delimiter=',')
    return pssm_matrix

def parse_pssm(_path, in_file, out_file):
    test_out = open(_path + out_file, 'w')
    isValid = True

    with open(_path + in_file) as fd:
        content = fd.readlines()[3:-6]
    
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for index, line in enumerate(content):
        line = line[6:-92].strip()                 # line[6:-92].strip() for only PSSM
        line = re.sub(r' +', ',', line)
        # csvLine = sequence[index] + ',' + line    # include label as first column
        csvLine = line

        # validity check
        cnt = csvLine.count(',')
        if cnt != 19:   # 20 for just PSSM, 42 for all, 19 for just PSSM and not include label
            isValid = False

        # write to csv files
        test_out.write(csvLine + '\n')

    test_out.close()

