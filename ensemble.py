import os
from process_result_problem_1 import CapsNet as CapsNet_1
from process_result_problem_2 import CapsNet as CapsNet_2
from keras import layers, models
from keras.models import Model


def load_trained_models_1(input_shape, directory):
    n_folds = 5
    models = list()
    for i in range(n_folds):
        # define model
        model = CapsNet_1(input_shape=input_shape, n_class=2, routings=3)
        weight_file = directory + '/fold_%d' % (i) + '/best_model.h5'
        model.load_weights(weight_file)
        print('load weight from ', weight_file)
        models.append(model)

    return models

def load_trained_models_2(input_shape, directory):
    n_folds = 10
    models = list()
    for i in range(n_folds):
        # define model
        model = CapsNet_2(input_shape=input_shape, n_class=2, routings=3)
        weight_file = directory + '/fold_%d' % (i) + '/best_model.h5'
        model.load_weights(weight_file)
        print('load weight from ', weight_file)
        models.append(model)

    return models

def ensemble_folds(models, input_shape):
    model_input = layers.Input(shape=input_shape)
    outputs = [model(model_input) for model in models]
    y = layers.Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model