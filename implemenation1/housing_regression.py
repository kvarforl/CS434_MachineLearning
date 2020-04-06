#!/usr/bin/env python3
import numpy as np

featuresX = np.loadtxt(open("housing_train.csv","rb"), delimiter=",", usecols=range(13))
featuresX = np.insert(featuresX, 0, [1], axis=1) #add dummy column of 1s
#print(featuresX.shape) #shape output is (rows, cols)

Y = np.loadtxt(open("housing_train.csv","rb"), delimiter=",", usecols=13)
#print(Y.shape)

w = np.linalg.inv(featuresX.T @ featuresX ) @ featuresX.T @ Y

print("Learned weight vector:", w)
