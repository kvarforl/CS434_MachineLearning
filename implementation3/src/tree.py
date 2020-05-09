import numpy as np

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None, forest=False, max_features=0, adaBoost=False):
		self.max_depth = max_depth
		self.forest = forest
		self.max_features = max_features
		self.adaBoost = adaBoost

	# take in features X and labels y
	# build a tree
	def fit(self, X, y, D=0):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1)

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth, ft_set="uninitialized"):
		num_samples, num_features = X.shape
		if(str(ft_set) == "uninitialized" and self.forest == True ):
			ft_set = np.arange(num_features)
		# which features we are considering for splitting on
		if(self.forest):
			self.features_idx = np.random.choice(ft_set, self.max_features, replace=False)
		else:
			self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature

			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
			#ft_set = ft_set[ft_set != best_feature]
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1, ft_set=ft_set)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1, ft_set=ft_set)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:

			########################################
			#       YOUR CODE GOES HERE            #
			########################################
			#assuming that every item in y is split into left or right (nothing remains uncategorized)
			pL = len(left_y) / len(y)
			pR = len(right_y) / len(y)
			gain = self._uncertainty(y) - (pL* self._uncertainty(left_y)) - (pR*self._uncertainty(right_y))
			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

	#assuming C means count in assignment description
	def _uncertainty(self, tlist):
		#need to think harder about what the list actually is, but here is me assuming its a bunch of 1s and 0s
		cpositive = np.count_nonzero(tlist == 1)
		cnegative = np.count_nonzero(tlist == 0)
		total = len(tlist)
		return 1 - ((cpositive/total)**2) - ((cnegative/total)**2)



class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

		self.trees = []
		for _ in range(self.n_trees):
			self.trees.append(DecisionTreeClassifier(self.max_depth, forest=True, max_features=max_features))


	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		print('Fitting Random Forest...\n')
		for i in range(self.n_trees):
			#print(i+1, end='\t\r')
			self.trees[i].fit(bagged_X[i], bagged_y[i])
			##################
			# YOUR CODE HERE #
			##################
		print()

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		for i in range(self.n_trees):
			#generate random sample
			num_samples, num_features = X.shape
			indices = np.random.choice(num_samples, num_samples, replace=True)
			samplesX = np.take(X,indices,axis=0)
			samplesY = np.take(y, indices, axis=0)
			bagged_X.append(samplesX)
			bagged_y.append(samplesY)
			#print("Shapes: X-", samplesX.shape, "Y-",samplesY.shape
		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)


	def predict(self, X):
		preds = []

		for i in range(self.n_trees):
			preds.append(np.array(self.trees[i].predict(X)))
		preds = np.array(preds)

		majority = np.count_nonzero(preds, axis=0).reshape(1, -1) #condense to 1d array of the number of 1s in each column
		majority[majority >= (0.5*self.n_trees)] = 1 #if the number of 1's is more than half, set to 1
		majority[majority != 1] = 0 #if it didn't get set to one in previous line, set to 0

		preds = majority.flatten()
		return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():
	"""
	AdaBoost Classifier. Build series of decision trees that will be iteritively based off the last one generated.

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, L):
		self.n_trees = L
		self.max_depth = 1

		self.trees = []
		self.dVectors = []
		self.alphaVector = []
		for _ in range(self.n_trees):
			self.trees.append(DecisionTreeClassifier(self.max_depth, adaBoost=True))
			self.alphaVector.append(1)
			self.dVectors.append(np.empty(2098))

	def fit(self, X, y):
		"""
		dWeights = np.empty(2098)
		self.dVectors
		dWeights.fill(1/2098)

		"""

		#bagged_X, bagged_y = self.bag_data(X, y)
		print('Fitting AdaBoost Descision Stumps...\n')

		# Initialize weights of first tree to uniform distribution
		self.dVectors[0].fill(1/2098)

		for i in range(self.n_trees):
			# Learn decision stump classifier with weight input
			#self.trees[i].fit(X[i], y[i], self.dVectors[i])

			# Calculate error of trained classifier
			#error = 1 - self.trees[i].accuracy_score(X, y)

			# Calculate alpha value
			#alphaVector[i] = log((1-error)/error)/2

			if (i < self.n_trees - 1): # The last stump won't calculate a new weight vector
				# Generate next weight vector
				# ...

				# Normalize weight vector
				#self.dVectors[i + 1] = self.dVectors[i + 1] / self.dVectors[i + 1].sum()

		print()


193