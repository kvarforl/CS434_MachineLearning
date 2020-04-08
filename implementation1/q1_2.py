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
#print(w.shape)

print("Learned weight vector:", w)

#not sure this is correct whoops; assuming yhat comes from x*w
training_ase = (1 / featuresX.shape[0]) * sum((y - yhat)**2 for y in Y for yhat in featuresX @ w)


testX = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=range(13))
testX = np.insert(testX, 0, [1], axis=1) #add dummy column of 1s
testY = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=13)

testing_ase = (1 / testX.shape[0]) * sum((y - yhat)**2 for y in testY for yhat in testX @ w)

print("Training ASE:",training_ase)
print("Testing ASE:", testing_ase)
