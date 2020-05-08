import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
	print('Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=20)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

	print("Root -- Feature ", clf.root.feature, "\tSplit:", clf.root.split)

def test_depth(x_train, y_train, x_test, y_test):
	print("Decision Tree Depth Test")
	depths = list(range(1, 26)) #1 through 25, inclusive
	test_accuracies = []
	train_accuracies = []
	f1_scores = []
	for d in depths:
		clf = DecisionTreeClassifier(max_depth=d)
		clf.fit(x_train, y_train)
		preds_train = clf.predict(x_train)
		preds_test = clf.predict(x_test)
		train_accuracies.append(accuracy_score(preds_train, y_train))
		test_accuracies.append(accuracy_score(preds_test, y_test))
		preds = clf.predict(x_test)
		f1_scores.append(f1(y_test, preds))
	print("done computing")
	fig, ax = plt.subplots()
	ax.plot(depths, test_accuracies)
	ax.plot(depths, train_accuracies)
	ax.set(xlabel="depth of decision tree", ylabel="accuracy", title="Accuracy vs Depth")
	fig.legend(["Testing", "Train"])
	fig.savefig("accuracies_v_tree_depth.png")

	fig, ax = plt.subplots()
	ax.plot(depths, f1_scores)
	ax.set(xlabel="depth of decision tree", ylabel="f1 score", title="F1 Score vs Depth")
	fig.savefig("f1scores_v_tree_depth.png")


def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

def ada_boost_testing(x_train, y_train, x_test, y_test):
	print('AdaBoost\n\n')
	abclf = AdaBoostClassifier(L=50)
	"""
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

	"""

	print("Initialized booster")





###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	if args.county_dict == 1:
		county_info(args)
	if args.decision_tree == 1:
		#decision_tree_testing(x_train, y_train, x_test, y_test)
		#test_depth(x_train, y_train, x_test, y_test)
		pass
	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)
	if args.ada_boost == 1:
		ada_boost_testing(x_train, y_train, x_test, y_test)


	print('Done')







