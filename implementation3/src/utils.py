import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import f1_score

def load_data(rootdir='./'):	
	x_train = np.loadtxt(rootdir+'x_train.csv', delimiter=',').astype(int)
	y_train = np.loadtxt(rootdir+'y_train.csv', delimiter=',').astype(int)
	x_test = np.loadtxt(rootdir+'x_test.csv', delimiter=',').astype(int)
	y_test = np.loadtxt(rootdir+'y_test.csv', delimiter=',').astype(int)
	y_train[y_train == -1] = 0
	y_test[y_test == -1] = 0
	return x_train, y_train, x_test, y_test

def load_dictionary(rootdir='./'):
	county_dict = pd.read_csv(rootdir+'county_facts_dictionary.csv')
	return county_dict

def dictionary_info(county_dict):
	for i in range(county_dict.shape[0]):
		print('Feature: {} - Description: {}'.format(i, county_dict['description'].iloc[i]))

def accuracy_score(preds, y):
	accuracy = (preds == y).sum()/len(y)
	return accuracy

def f1(y, yhat):
	return f1_score(y, yhat)



###########################################################################
# you may add plotting or data processing functions (etc) in here if desired
###########################################################################


