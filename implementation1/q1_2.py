#!/usr/bin/env python3

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("training", help="a .csv file of training data")
parser.add_argument("test_data", help="a .csv file of test data")
args = parser.parse_args()


featuresX = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=range(13))
featuresX = np.insert(featuresX, 0, [1], axis=1) #add dummy column of 1s
#print(featuresX.shape) #shape output is (rows, cols)

Y = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=13)
#print(Y.shape)

w = np.linalg.inv(featuresX.T @ featuresX ) @ featuresX.T @ Y

print("Learned weight vector:", w)
