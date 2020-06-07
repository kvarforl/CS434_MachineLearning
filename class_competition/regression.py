from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
import argparse
import pandas as pd
import numpy as np

submission_for_kaggle = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="a .csv file of training data")
    parser.add_argument("test", help="a .csv file of test data")
    return parser.parse_args()

def load_data():
    if (submission_for_kaggle):
        # These next two lines are specifically for the kaggle notebook
        train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv",sep="," ).astype("str")
        test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv",sep=",").astype("str")
    else:
        args = get_args()
        #for some reason the train["text"][314] coms in as nan. we are making it 'nan' for now with astype
        train = pd.read_csv(args.train,sep=",").astype("str")
        test = pd.read_csv(args.test,sep=",").astype("str")

    return train, test

def _clean_text(review):

    review = str(review)
    #omit URLS
    review = re.sub(r'^https?:\/\/.*[\r\n]*', '', review)

    #ignore case 
    review = review.lower()
    #ignore stopwords that are not negations
    #review = [w for w in review if not w in stop_words]
    return review

#trainX is bow, train is pandas df
#returns best estimator
def model_selection(trainX, train):
    print("grid searching...")
    #tuning parameters of LogisticRegression with cross validation using GridSerachCV
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(trainX, train["sentiment"])

    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    return grid.best_estimator_

print("loading data..")
alltrain, alltest = load_data()

#remove neutral examples for log regression classifier
train = alltrain[alltrain["sentiment"]!="neutral"]
test = alltest[alltest["sentiment"] != "neutral"]

#params and ideas for general reg from https://itnext.io/machine-learning-sentiment-analysis-of-movie-reviews-using-logisticregression-62e9622b4532
#doesn't currently clean text at all; can add as param to CountVectorizer
vec = CountVectorizer(min_df=1, 
                      ngram_range=(2,2), 
                      preprocessor=_clean_text)
#training model on Tweets, and bagging words

#puking and barfing if we get anything thats nan or null, should have been transformed to "" or "nan"
if(train["text"].isnull().values.any()):
    print("found the nan! yikes. stopping")
    quit()

print("bagging data...")
trainX = vec.fit(train["text"]).transform(train["text"])
testX = vec.transform(test["text"])

#lr = model_selection(trainX, train)
#from model selection, just speeding up the runtime. change when the data changes
print("fitting regression model")
lr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
lr.fit(trainX, train["sentiment"])

#now use vocabulary and coef_ indices to find how pos or neg a word is
vocab = dict([[v,k] for k,v in vec.vocabulary_.items()])
#vocab is now structured as indices_of_coeff: word(s)


