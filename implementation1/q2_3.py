#!/usr/bin/env python3

import numpy as np
import math as math
import matplotlib
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="a .csv file of training data")
    parser.add_argument("test_data", help="a .csv file of test data")
    parser.add_argument("lambdas", type=float)
    return parser.parse_args()


def make_plot(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="lambda",
            ylabel = "accuracy",
            title=title)
    #plt.xticks(np.arange(0, len(lambdas), (max(lambdas) - min(lambdas))/len(lambdas)));
    fig.savefig(title.replace(" ", "_")+".png")
    plt.show()


def calc_prediction(w, x):
    #outcome = w.T @ x.reshape(256, 1)
    outcome = calc_sigmoid(x, w)
    #print("outcome: ", outcome)
    if (outcome > 0.5):
        return True
    else:
        return False


def calc_accuracy(X, Y, w):
    matches = 0
    for i in range(len(X)):
        if(calc_prediction(w, X[i]) == Y[i]):
                matches += 1
    print("matches calculated: ", matches, " out of ", len(Y))
    if (matches == 0):
        if (len(Y) == 0):
            return 1
        else:
            return 0
    else:
        return matches / len(Y)


def calc_sigmoid(featureVector, w):
    exponential = float(-w.T @ featureVector.reshape(256, 1))
    return 1 / (1 + math.exp(exponential))

def calc_regularization(w, lambdaValue):
    return 0.5 * lambdaValue * np.square(np.linalg.norm(w))

def calc_w(trainX, trainY, testX, testY, lr, lambdas):
    gradientDecentIterations = 20
    trainAccuracies = []
    testAccuracies = []
    for l in range(len(lambdas)):
        w = np.zeros((256 , 1)) # initialize weight vector
        for _ in range(gradientDecentIterations):
            gradient = np.zeros((256, 1)) # initialize gradient vector
            for i in range(len(trainY)):
                sigmoid = calc_sigmoid(trainX[i], w)
                #print("sigmoid: " + str(sigmoid))
                gradient = gradient + (sigmoid - trainY[i]) * trainX[i].reshape(256, 1) + calc_regularization(w, lambdas[l])

            w = w - lr * gradient
        trainAccuracies.append(calc_accuracy(trainX, trainY, w))
        testAccuracies.append(calc_accuracy(testX, testY, w))

    epochArray = list(range(0, gradientDecentIterations))
    print((trainAccuracies))
    make_plot(lambdas, trainAccuracies, "Training Accuracy")
    print((testAccuracies))
    make_plot(lambdas, testAccuracies, "Testing Accuracy")
    return w



args = get_args()

lr = 0.0000001 # Learning rate
lambdas = args.lambdas

lambdas = []

for i in range(int(args.lambdas)):
    lambdas.append(float(input("Enter lambda value #" + str(i) + ": ")))

trainX = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=range(256))
#print(trainX.shape) #shape output is (rows, cols)

trainY = np.loadtxt(open(args.training,"rb"), delimiter=",", usecols=256)
#print(trainY.shape)

testX = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=range(256))
#print(testX.shape)

testY = np.loadtxt(open(args.test_data,"rb"), delimiter=",", usecols=256)
#print(testY.shape)

# normalize data
#trainX /= 255
#trainY /= 255

w = calc_w(trainX, trainY, testX, testY, lr, lambdas)

#print("Learned weight vector:", w)

trainingAccuracy = calc_accuracy(trainX, trainY, w)
print("Final Training Accuracy: ", trainingAccuracy)

testingAccuracy = calc_accuracy(testX, testY, w)
print("Final Testing Accuracy: ", testingAccuracy)
