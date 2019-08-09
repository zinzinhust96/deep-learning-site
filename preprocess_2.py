"""FASTA preprocessing."""

import os
import math
# import numpy for caculating
import numpy as np
# import Bio for fasta file handling
from Bio import SeqIO
# import KFold for cross-validation handling
from sklearn.model_selection import KFold

# 5-fold cross-validation
KF = KFold(n_splits=5)
# window size for sliding window technique
WINDOW_SIZE = 21
# 20 standard amino acids presentation and '@' digit present the missing residue
STANDARD_AMINO_ACID = 'ARNDCQEGHILKMFPSTWYV@'
# convert standard amino acids char list to int list
char_to_int = dict((c, i) for i, c in enumerate(STANDARD_AMINO_ACID))
# convert standard amino acids int list to char list
int_to_char = dict((i, c) for i, c in enumerate(STANDARD_AMINO_ACID))
# number of classes
NUM_CLASSES = 2

class WindowSlidePSSMExtractor:
    def __init__(self, sequence):

        # Rescale the value
        sequence = 1 / (1 + np.exp(-sequence))

        # padding zero vector to sequence
        pad_size = int(WINDOW_SIZE / 2)
        # print('SEQUENCE SHAPE: ', sequence.shape)
        sequence = np.pad(sequence, [(pad_size, pad_size), (0, 0)], mode='constant', constant_values=0)

        # feature sequence
        residue_feature_sequence = self.get_residue_feture_sequence(sequence)

        # add the attributes to self
        self.sequence = sequence
        self.features = residue_feature_sequence

    # convert to final feature sequence base on sliding-window technique
    @staticmethod
    def get_residue_feture_sequence(pssm_encoded_sequence):
        # init an empty list
        residue_feature_sequence = list()
        # loop through the sequence
        for i in range(len(pssm_encoded_sequence) - WINDOW_SIZE + 1):
            # each element is converted to [WINDOW_SIZE] elements around it
            residue_feature = pssm_encoded_sequence[i: i + WINDOW_SIZE]
            # append to the final sequence
            residue_feature_sequence.append(residue_feature)
        return residue_feature_sequence

class OneHotExtractorWithPadding:
    def __init__(self, name, sequence):

        # append missing residue element to sequence
        for _ in range(int(WINDOW_SIZE / 2)):
            sequence = "@" + sequence + "@"

        # integer encode the sequence
        integer_encoded_seq = self.get_integer_values_of_sequence(sequence)

        # one hot the sequence
        # onehot_encoded_seq = self.get_one_hot_sequence(integer_encoded_seq)

        # feature sequence
        residue_feature_sequence = self.get_residue_feture_sequence(integer_encoded_seq)

        # add the attributes to self
        self.name = name
        self.sequence = sequence
        self.integer = integer_encoded_seq
        # self.onehot = onehot_encoded_seq
        self.features = residue_feature_sequence

    # get integer values from a sequence
    @staticmethod
    def get_integer_values_of_sequence(sequence):
        integer_encoded = [char_to_int[char] for char in sequence]
        return integer_encoded

    # one-hot encoding an integer sequence
    @staticmethod
    def get_one_hot_sequence(integer_sequence):
        # init an empty list
        one_hot_encoded_sequence = list()
        # loop through integer sequence and flag the index of each amino acid as 1 for each amino acid
        for value in integer_sequence:
            one_hot_encoded_char = [0 for _ in range(len(STANDARD_AMINO_ACID))]
            one_hot_encoded_char[value] = 1
            one_hot_encoded_sequence.append(one_hot_encoded_char)
        return one_hot_encoded_sequence

    # convert one-hot encoded sequence to final feature sequence base on sliding-window technique
    @staticmethod
    def get_residue_feture_sequence(one_hot_encoded_sequence):
        # init an empty list
        residue_feature_sequence = list()
        # loop through the sequence
        for i in range(len(one_hot_encoded_sequence) - WINDOW_SIZE + 1):
            # each element is converted to [WINDOW_SIZE] elements around it
            residue_feature = one_hot_encoded_sequence[i: i + WINDOW_SIZE]
            # append to the final sequence
            residue_feature_sequence.append(residue_feature)
        return residue_feature_sequence


class OneHotExtractorWithoutPadding:
    def __init__(self, name, sequence):
        seq_length = len(sequence)

        # append missing residue element to sequence
        for i in range(int(WINDOW_SIZE / 2)):
            sequence = sequence[(seq_length - 1) - (i % seq_length)] + sequence + sequence[i % seq_length]

        # integer encode the sequence
        integer_encoded_seq = self.get_integer_values_of_sequence(sequence)

        # one hot the sequence
        # onehot_encoded_seq = self.get_one_hot_sequence(integer_encoded_seq)

        # feature sequence
        residue_feature_sequence = self.get_residue_feture_sequence(integer_encoded_seq)

        # add the attributes to self
        self.name = name
        self.sequence = sequence
        self.integer = integer_encoded_seq
        # self.onehot = onehot_encoded_seq
        self.features = residue_feature_sequence

    # get integer values from a sequence
    @staticmethod
    def get_integer_values_of_sequence(sequence):
        integer_encoded = [char_to_int[char] for char in sequence]
        return integer_encoded

    # one-hot encoding an integer sequence
    @staticmethod
    def get_one_hot_sequence(integer_sequence):
        # init an empty list
        one_hot_encoded_sequence = list()
        # loop through integer sequence and flag the index of each amino acid as 1 for each amino acid
        for value in integer_sequence:
            one_hot_encoded_char = [0 for _ in range(len(STANDARD_AMINO_ACID))]
            one_hot_encoded_char[value] = 1
            one_hot_encoded_sequence.append(one_hot_encoded_char)
        return one_hot_encoded_sequence

    # convert one-hot encoded sequence to final feature sequence base on sliding-window technique
    @staticmethod
    def get_residue_feture_sequence(one_hot_encoded_sequence):
        # init an empty list
        residue_feature_sequence = list()
        # loop through the sequence
        for i in range(len(one_hot_encoded_sequence) - WINDOW_SIZE + 1):
            # each element is converted to [WINDOW_SIZE] elements around it
            residue_feature = one_hot_encoded_sequence[i: i + WINDOW_SIZE]
            # append to the final sequence
            residue_feature_sequence.append(residue_feature)
        return residue_feature_sequence


# get the test features from input file path
def get_test_features(input_file):
    # init the empty list
    input_feature = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        # get the ResidueFeatureExtractor object of current sequence
        extractor = OneHotExtractorWithPadding(name, sequence)
        # append the feature to the list
        input_feature += extractor.features
    # convert to array
    return np.array(input_feature, dtype="float32")

# get the test labels from input file path
def get_test_labels(input_file):
    # init the empty list
    input_labels = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get the value of each sequence
        sequence = str(fasta.seq)
        for value in list(sequence):
            one_hot_encoded_vector = [0 for _ in range(NUM_CLASSES)]
            one_hot_encoded_vector[int(value)] = 1
            # append to the list
            input_labels.append(one_hot_encoded_vector)

    # convert to array
    return np.array(input_labels, dtype="float32")

def get_pssm_data(input_folder):
    # init the empty list
    input_features = list()
    input_labels = list()
    # list all csv files in that folder
    pssm_files = os.listdir(input_folder)
    # loop all the files
    for pssm_file in pssm_files:
        matrix = np.genfromtxt(input_folder + pssm_file, delimiter=',')
        labels = matrix[:, 0]
        features = matrix[:, 1:]

        # append to labels list
        for value in labels:
            one_hot_encoded_vector = [0 for _ in range(NUM_CLASSES)]
            one_hot_encoded_vector[int(value)] = 1
            # append to the list
            input_labels.append(one_hot_encoded_vector)

        # append to feature list
        # get the WindowSlidePSSMExtractor object of current sequence
        extractor = WindowSlidePSSMExtractor(features)
        # append the feature to the list
        input_features += extractor.features

    # convert to array
    return np.array(input_features, dtype="float32"), np.array(input_labels, dtype="float32")

def get_train_and_validation_data(x, y, KF, cross_val_index):
    # split train-test set following k-fold cross validation
    splited_features = list(KF.split(x, np.argmax(y, axis=1)))
    # get the training set index
    train_index = splited_features[cross_val_index][0]
    # get the evaluating set index
    eval_index = splited_features[cross_val_index][1]
    print('EVAL INDEX: ', eval_index)
    print("Train Data Ratio: ", sum(np.argmax(y[train_index], axis=1)==0) / sum(np.argmax(y[train_index], axis=1)==1))
    print("Test Data Ratio: ", sum(np.argmax(y[eval_index], axis=1)==0) / sum(np.argmax(y[eval_index], axis=1)==1))

    # return training and evaluate set
    return (x[train_index], y[train_index]), (x[eval_index], y[eval_index])