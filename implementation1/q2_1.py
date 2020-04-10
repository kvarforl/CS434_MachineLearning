#!/usr/bin/env python3

import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="a .csv file of training data")
    parser.add_argument("test_data", help="a .csv file of test data")
    return parser.parse_args()


args = get_args()

trainX = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=range(256))
#print(trainX.shape) #shape output is (rows, cols)

trainY = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=256)
#print(trainY.shape)

testX = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=range(256))
testY = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=256)
#print(testX.shape)
#print(testY.shape)

