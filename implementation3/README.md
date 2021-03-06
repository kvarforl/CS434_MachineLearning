This project explores different aspect of using a Decision Tree Ensemble for Predicting Election Results by US County Statistics. Starter code was provided by Taylor Dinkins.
To run:

```bash
cd src
python3 main.py
```

Code for forest, tree, and Adaboost generation can be found in src/tree.py. src/main.py includes functions for generating plots to consider the impacts of tree depth, number of trees in a forest, and max features to consider in forest building. Note that this data is class imbalanced and discretized; we did not adjust our implementation to address this, but we did have class discussion and reading on different methods of mitigation.

Code can also be run for specific classifier generation and testing ->

Decision Tree Classifier:
```bash
python3 main.py --decision_tree=1 --random_forest=0 --ada_boost=0
```

Random Forest with Bagging Classifier :
```bash
python3 main.py --decision_tree=0 --random_forest=1 --ada_boost=0
```

Decision Stump Ensemble with Adaboost Classifier:
```bash
python3 main.py --decision_tree=0 --random_forest=0 --ada_boost=1
```
