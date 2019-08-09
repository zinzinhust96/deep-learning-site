"""FASTA preprocessing."""

# import numpy for caculating
import numpy as np
# import Bio for fasta file handling
from Bio import SeqIO
# import tensorflow 
import tensorflow as tf
# import KFold for cross-validation handling
from sklearn.model_selection import KFold

# 5-fold cross-validation
KF = KFold(n_splits=5)

DINUCLEOTIDE = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU', 'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']
# convert standard amino acids char list to int list
char_to_int = dict((c, i) for i, c in enumerate(DINUCLEOTIDE))
# convert standard amino acids int list to char list
int_to_char = dict((i, c) for i, c in enumerate(DINUCLEOTIDE))


# Residue Feature Extractor
class ResidueFeatureExtractor:
    def __init__(self, name, sequence):

        # dinucleotide statistic
        dinucleotide_seq, dinucleotide_stat = self.get_dinucleotide_stat_and_sequence(sequence)

        # frequency-hot the sequence
        frequency_hot_encoded_seq = self.get_frequency_hot_sequence(dinucleotide_seq, dinucleotide_stat)

        # add the atdibutes to self
        self.name = name
        self.sequence = sequence
        self.features = frequency_hot_encoded_seq

    # extract dinucleotide sequence from nucleotide sequence
    @staticmethod
    def get_dinucleotide_stat_and_sequence(sequence):
        dinucleotide_seq = list()
        dinucleotide_stat = [0 for _ in range(len(DINUCLEOTIDE))]
        for i in range(len(sequence)-1):
            dinuc = sequence[i:i+2]
            dinucleotide_stat[char_to_int[dinuc]] += 1
            dinucleotide_seq.append(dinuc)

        return dinucleotide_seq, dinucleotide_stat

    # one-hot encoding an integer sequence
    @staticmethod
    def get_frequency_hot_sequence(char_sequence, char_stat):
        # init an empty list
        frequency_hot_encoded_sequence = list()
        # loop through integer sequence and flag the index of each amino acid as 1 for each amino acid
        for value in char_sequence:
            frequency_hot_encoded_char = [0 for _ in range(len(DINUCLEOTIDE))]
            int_value = char_to_int[value]
            frequency_hot_encoded_char[int_value] = 1
            frequency_hot_encoded_sequence.append(frequency_hot_encoded_char)
        return frequency_hot_encoded_sequence

# get input data from input file path, follow the cross-validation index
def get_input_data(input_file, cross_val_index):
    # init an empty list
    input_features = list()
    input_labels = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        # get the ResidueFeatureExtractor object of current sequence
        extractor = ResidueFeatureExtractor(name, sequence)
        # append the feature to the list
        input_features.append(extractor.features)
        # check the name of each sequence and append the label to the list
        if name[0] == 'P':
            one_hot_encoded_vector = [0, 1]
        else:
            one_hot_encoded_vector = [1, 0]
        input_labels.append(one_hot_encoded_vector)

    # convert to array with data type uint8 (0-255)
    input_features = np.array(input_features, dtype="float32")
    input_labels = np.array(input_labels, dtype="float32")
    # split train-test set following k-fold cross validation
    splited_features = list(KF.split(input_features))
    # get the training set index
    train_index = splited_features[cross_val_index][0]
    # get the evaluating set index
    eval_index = splited_features[cross_val_index][1]
    
    # return training and evaluate set
    return input_features[train_index], input_labels[train_index] , input_features[eval_index], input_labels[eval_index]

# get the test data from input file path
def get_test_data(input_file):
    # init the empty list
    input_features = list()
    input_labels = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        # get the ResidueFeatureExtractor object of current sequence
        extractor = ResidueFeatureExtractor(name, sequence)
        # append the feature to the list
        input_features.append(extractor.features)
        # check the name of each sequence and append the label to the list
        if name[0] == 'P':
            one_hot_encoded_vector = [0, 1]
        else:
            one_hot_encoded_vector = [1, 0]
        input_labels.append(one_hot_encoded_vector)
    
    # convert to array with data type uint8 (0-255)
    input_features = np.array(input_features, dtype="float32")
    input_labels = np.array(input_labels, dtype="float32")
    return input_features, input_labels
