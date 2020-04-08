#!/usr/bin/env python3

import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="a .csv file of training data")
    parser.add_argument("test_data", help="a .csv file of test data")
    return parser.parse_args()

#x and y are of type np.ndarray
def calc_w(x, y):
    return np.linalg.inv(x.T @ x ) @ x.T @ y

#assumes that yhat comes from x*w
def calc_ase(x, y, w):
    return (1 / x.shape[0]) * sum((yi - yhati)**2 for yi in y for yhati in x @ w)

#returns features matrix with num columns of np.random.normal appended
def append_cols(features, num):
    return_mat = features.copy()
    rows, cols = return_mat.shape
    for _ in range(num):
        return_mat = np.insert(return_mat, cols, [np.random.normal(size=rows)], axis=1)
    return return_mat

args = get_args()

featuresX = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=range(13))
featuresX = np.insert(featuresX, 0, [1], axis=1) #add dummy column of 1s
Y = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=13)
w = calc_w(featuresX, Y)

training_entry_info = {
    "features": featuresX,
    "weights": w,
    "ase" : calc_ase(featuresX, Y, w)
}

#key = d, value = (matrix with d values, learned w from matrix, ase)
training_data = { 0: training_entry_info}
for i in range(2,20, 2):
    training_entry_info = {}
    matrix = append_cols(training_data[i-2]["features"], 2)#add 2 random cols to previous random matrix
    weights = calc_w(matrix, Y)
    ase = calc_ase(matrix, Y, weights)
    training_data[i] = {"features": matrix,
                        "weights": weights,
                        "ase": ase,
                        }

testX = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=range(13))
testX = np.insert(testX, 0, [1], axis=1) #add dummy column of 1s
testY = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=13)
testing_ase = calc_ase(testX, testY, w)

testing_entry_info = {
        "features": testX,
        "ase": testing_ase
        }

#key = d, value = (matrix with d values,ase)
testing_data = { 0: testing_entry_info}
for i in range(2,20, 2):
    matrix = append_cols(testing_data[i-2]["features"], 2)#add 2 random cols to previous random matrix
    ase = calc_ase(matrix, Y, training_data[i]["weights"]) #use training weight vector
    testing_data[i] = {"features": matrix,
                        "ase" : ase
                        }


