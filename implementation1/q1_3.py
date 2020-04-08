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


args = get_args()

featuresX = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=range(13))
#print(featuresX.shape) #shape output is (rows, cols)

Y = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=13)
#print(Y.shape)

w = calc_w(featuresX, Y)
#print(w.shape)

print("Learned weight vector:", w)

#not sure this is correct whoops; assuming yhat comes from x*w
training_ase = calc_ase(featuresX, Y, w)


testX = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=range(13))
testY = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=13)

testing_ase = calc_ase(testX, testY, w)

print("Training ASE:",training_ase)
print("Testing ASE:", testing_ase)
