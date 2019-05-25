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
from preprocess1 import ResidueFeatureExtractor

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

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv1')(x)
    #conv1=BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.7)(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, kernel_initializer='he_normal', strides=2, padding='valid', dropout=0.2)

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
    models = list()
    for i in range(5):
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

def process_result_problem_1(fasta_seq, dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # read fasta
    name, sequence = fasta_seq.split('\n')
    features = ResidueFeatureExtractor(name, sequence).features
    features = np.array(features, dtype="float32")
    # add one shape to sequence list for standardizing 
    new_shape = list(features.shape)
    new_shape.insert(0, 1)
    new_shape.append(1)
    features = np.reshape(features, new_shape)

    # load model weights based on dataset type
    dataset_weight = ''
    if dataset == '1':    # H. sapiens
        dataset_weight = dir_path + '/problem-1/seed_11'
    else:               # S. cerevisiae
        dataset_weight = dir_path + '/problem-1/seed_13'
    
    print(features.shape)
    trained_models = load_trained_models(input_shape = features.shape[1:], directory = dataset_weight)
    ensemble_model = ensemble(trained_models, input_shape=features.shape[1:])
    y_pred = ensemble_model.predict(features)
    y_pred_prob = calculate_probability(y_pred)
    y_pred_label = np.where(np.array(y_pred_prob) < 0.5, 0, 1)
    return y_pred_label[0]
