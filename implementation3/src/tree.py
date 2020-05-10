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
	def fit(self, X, y, D="uninitialized"):
		self.classes = list(set(y))
		if (self.adaBoost):
			self.root = self.build_tree(X, y, depth=1, D=D)
		else:
			self.root = self.build_tree(X, y, depth=1)

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		if(self.adaBoost):
			if example[node.feature] < node.split:
				return -1
			else:
				return 1
		else:
			while node.left_tree:
				if example[node.feature] < node.split:
					node = node.left_tree
				else:
					node = node.right_tree
		if(self.adaBoost and node.prediction == 0):
			return -1
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth, ft_set="uninitialized", D="uninitialized"):
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
		num_samples_per_class = [np.sum(y == i) for i in self.classes]
		prediction = np.argmax(num_samples_per_class)

		# if(self.adaBoost):
		# 	sample_weights = [np.sum(D[y==i]) for i in self.classes]
		# 	print("predicts index of larger sample weight")
		# 	print("sample_weights:", sample_weights)
		# 	prediction = np.argmax(sample_weights)
		# 	print("prediction", prediction)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature

			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split, D)
					
					# if we have a better gain, use this split and keep track of data
					# print("gain: ", gain)
					if gain > best_gain:
						#print("new best gain: ", gain)
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y

		#need to predict based on best gain; use weights at D[best_feature] ?? yikes
		# if we haven't hit a leaf node
		# add subtrees recursively
		if (not(self.adaBoost)):
			if best_gain > 0.0:
				left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1, ft_set=ft_set)
				right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1, ft_set=ft_set)
				return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)


		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split, D="uninitialized"):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]

		if (self.adaBoost):
			# calculate benefit of split ()
			left_d = D[left_idx]
			right_d = D[right_idx]
			gain = self.calculate_weighted_gain(y, left_y, right_y, left_d, right_d, D)
		else:
			# calculate gini impurity and gain for y, left_y, right_y
			gain = self.calculate_gini_gain(y, left_y, right_y)
			
		
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			#assuming that every item in y is split into left or right (nothing remains uncategorized)
			pL = len(left_y) / len(y)
			pR = len(right_y) / len(y)
			gain = self._uncertainty(y) - (pL* self._uncertainty(left_y)) - (pR*self._uncertainty(right_y))
			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

	def calculate_weighted_gain(self, y, left_y, right_y, left_d, right_d, d):
		# not a leaf node
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			#assuming that every item in y is split into left or right (nothing remains uncategorized)
			pL = np.sum(left_d) / np.sum(d)
			pR = np.sum(right_d) / np.sum(d)
			gain = self._weighted_uncertainty(y, d) - (pL* self._weighted_uncertainty(left_y,left_d)) - (pR*self._weighted_uncertainty(right_y, right_d))
			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

	def _weighted_uncertainty(self, tlist, dlist):
		cpositive = np.sum(dlist[tlist == 1])#sum the weights at indices where t == 1
		cnegative = np.sum(dlist[tlist == 0])
		total = np.sum(dlist)
		return 1 - ((cpositive/total)**2) - ((cnegative/total)**2)


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
			self.trees[i].fit(bagged_X[i], bagged_y[i])
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

		print('Fitting AdaBoost Descision Stumps...\n')

		# Initialize weights of first tree to uniform distribution
		self.dVectors[0].fill(1/2098)

		for i in range(self.n_trees):
			# Learn decision stump classifier with weight input
			self.trees[i].fit(X, y, self.dVectors[i])

			# Calculate error of trained classifier
			preds = np.array(self.trees[i].predict(X)).astype("float64").copy()
			error = np.sum(self.dVectors[i][preds !=y])
			#error = 1 - self.trees[i].accuracy_score(X, y)
			print("Error: ", error)

			if(error == 0.0):
				self.alphaVector[i] = 500.0
			else:	
				# Calculate alpha value
				self.alphaVector[i] = np.log((1-error)/error)/2
			print("Alpha: ", self.alphaVector[i])

			if (i < self.n_trees - 1): # The last stump won't calculate a new weight vector
				# Generate next weight vector
				m_factor = np.array(self.trees[i].predict(X)).astype("float64")
				#m_factor is all -1
				correct = np.exp(self.alphaVector[i])
				incorrect = np.exp(-1*self.alphaVector[i])
				m_factor[m_factor == y] = correct 
				m_factor[m_factor != correct] = incorrect
				
				self.dVectors[i+1] = np.multiply(self.dVectors[i], m_factor)

				# Normalize weight vector
				self.dVectors[i + 1] = self.dVectors[i + 1] / self.dVectors[i + 1].sum()
				print()

		print()

	#breaking because of error issue: otherwise should be good?
	def predict(self, X):
		sum_vector = np.zeros((X.shape[0])) #initialize sum vector to 0, len num ex
		for i in range(self.n_trees):
			preds = np.array(self.trees[i].predict(X)).astype("float64").copy() #a prediction vector
			preds = preds * self.alphaVector[i]
			sum_vector = sum_vector + preds#accumulate
		return np.sign(sum_vector) #return signs of pred vector
