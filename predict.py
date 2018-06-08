# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np


def hamming_distance(x, x_train):
    distance_matrix = np.zeros(shape=(len(x[:, 0]), len(x_train[:, 0])))
    for i in range(0, len(x[:, 0])):
        for k in range(0, len(x_train[:, 0])):
            distance_matrix[i, k] = np.count_nonzero(x[i, :] != x_train[k, :])
    return distance_matrix


def predict(x):
    size = 6050
    data = pkl.load(open('train.pkl', mode='rb'))
    return selection_knn(x, data[0][0:size], data[1][0:size])


def predict(x, size):
    data = pkl.load(open('train.pkl', mode='rb'))
    return selection_knn(x, data[0][0:size], data[1][0:size])


def selection_knn(x, x_train, y_train):
    predicted = np.zeros(shape=(len(x[:, 0]), 1))
    distance_matrix = hamming_distance(x, x_train)
    for k in range(0, len(x)):
        predicted[k, 0] = y_train[np.argmin(distance_matrix[k, :])]
    return predicted
