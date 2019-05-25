import numpy as np
import matplotlib.pyplot as plt

indexes = np.arange(0, 0.5, 0.0001)

def calculate_probability(y_pred):
    y_pred_prob = []
    for i in y_pred:
        y_pred_prob.append(i[1] / (i[0] + i[1]))

    return y_pred_prob