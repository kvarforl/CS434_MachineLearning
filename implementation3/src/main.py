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



def test_num_trees(x_train, y_train, x_test, y_test):
	print("Decision Tree Depth Test")
	n_trees = list(range(10, 210,10)) #[10,20,...200]
	test_accuracies = []
	train_accuracies = []
	f1_scores = []
	for t in n_trees:
		rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=t)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		train_accuracies.append(accuracy_score(preds_train, y_train))
		test_accuracies.append(accuracy_score(preds_test, y_test))
		preds = rclf.predict(x_test)
		f1_scores.append(f1(y_test, preds))
	print("done computing")
	fig, ax = plt.subplots()
	ax.plot(n_trees, test_accuracies)
	ax.plot(n_trees, train_accuracies)
	ax.set(xlabel="number of trees in forest", ylabel="accuracy", title="Accuracy vs NumTrees")
	fig.legend(["Testing", "Train"])
	plt.show()
	fig.savefig("accuracies_v_num_trees.png")	

	fig, ax = plt.subplots()
	ax.plot(n_trees, f1_scores)
	ax.set(xlabel="number of trees in forest", ylabel="f1 score", title="F1 Score vs NumTrees")
	plt.show()
	fig.savefig("f1scores_v_num_trees.png")	

def test_max_features(x_train, y_train, x_test, y_test):
	print("Decision Tree Depth Test")
	max_features = [1,2,5,8,10,20,25,35,50]
	test_accuracies = []
	train_accuracies = []
	f1_scores = []
	for m in max_features:
		rclf = RandomForestClassifier(max_depth=7, max_features=m, n_trees=50)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		train_accuracies.append(accuracy_score(preds_train, y_train))
		test_accuracies.append(accuracy_score(preds_test, y_test))
		preds = rclf.predict(x_test)
		f1_scores.append(f1(y_test, preds))
	print("done computing")
	fig, ax = plt.subplots()
	ax.plot(max_features, test_accuracies)
	ax.plot(max_features, train_accuracies)
	ax.set(xlabel="max number of features", ylabel="accuracy", title="Accuracy vs MaxFeatures")
	fig.legend(["Testing", "Train"])
	plt.show()
	fig.savefig("accuracies_v_maxfeatures.png")	

	fig, ax = plt.subplots()
	ax.plot(max_features, f1_scores)
	ax.set(xlabel="max number of features", ylabel="f1 score", title="F1 Score vs MaxFeatures")
	plt.show()
	fig.savefig("f1scores_v_maxfeatures.png")	

def run_trials(x_train, y_train, x_test, y_test):
	test_accuracies = []
	train_accuracies = []
	f1_scores = []
	for _ in range(10):
		rclf = RandomForestClassifier(max_depth=7, max_features=25, n_trees=130)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		train_accuracies.append(accuracy_score(preds_train, y_train))
		test_accuracies.append(accuracy_score(preds_test, y_test))
		preds = rclf.predict(x_test)
		f1_scores.append(f1(y_test, preds))

	print("Test Accuracies", "Train Accuracies", "F1 Scores", sep="\t")
	for i in range(10):	
		print(test_accuracies[i], train_accuracies[i],f1_scores[i], sep="\t")
	print()

	print(np.mean(test_accuracies), np.mean(train_accuracies), np.mean(f1_scores), sep="\t")	

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
		#run tests and generate graphs
		#test_num_trees(x_train, y_train, x_test, y_test) 
		#test_max_features(x_train, y_train, x_test, y_test)
		#run_trials(x_train, y_train, x_test, y_test)
	if args.ada_boost == 1:
		ada_boost_testing(x_train, y_train, x_test, y_test)


		
	print('Done')







