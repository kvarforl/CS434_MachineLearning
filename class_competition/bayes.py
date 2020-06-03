#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
from string import punctuation
from itertools import combinations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="a .csv file of training data")
    parser.add_argument("test", help="a .csv file of test data")
    return parser.parse_args()

#returns data as pandas dataframes: use test.head() to easily inspect
def load_data():
    args = get_args()
    train = pd.read_csv(args.train,sep="," )
    test = pd.read_csv(args.test,sep=",")
    return train, test

#function takes in a space delimited string and returns a cleaned list of words
#presently does not omit numbers (but it easily could)
def _clean_text(review):
    translator = str.maketrans('','',punctuation)
    return np.array(str(review).translate(translator).lower().split())

#takes in pandas df of test and train data
def clean_train_data(train):
    pos = train.loc[train["sentiment"] == "positive"].to_numpy() #create np matrix out of pos rows
    neg = train.loc[train["sentiment"] == "negative"].to_numpy() #create np matrix out of neg rows

    _, posTweets, posSelectedTxt, _ = pos.T
    _, negTweets, negSelectedTxt, _ = neg.T

    posVocab = np.unique(_clean_text(list(posSelectedTxt)))
    negVocab = np.unique(_clean_text(list(negSelectedTxt)))

    cT = [_clean_text(x) for x in posTweets]
    cS = [_clean_text(x) for x in posSelectedTxt]
    posTweets = np.array(cT)
    posSelectedTxt = np.array(cS)

    cT = [_clean_text(x) for x in negTweets]
    cS = [_clean_text(x) for x in negSelectedTxt]
    negTweets = np.array(cT)
    negSelectedTxt = np.array(cS)

    #all text fields are now a jagged array of cleaned examples
    return (posTweets, posSelectedTxt), (negTweets, negSelectedTxt), posVocab, negVocab

#really just for assignment specifications; outputs comma delimited text file of preprocess results
def output_info(bow, labels, vocab, filename):
    with open(filename,"w") as fp:
        print(*vocab, "classlabel", sep=",", file=fp)
        n_examples = bow.shape[0]
        for ind in range(n_examples):
            if(labels != []):
                print(*bow[ind], labels[ind], sep=",", file=fp)
            else:
                print(*bow[ind], sep=",", file=fp)

#expects 1D np arrays of equal dimensions
def accuracy_score(preds, labels):
    correct = np.count_nonzero(preds==labels)
    return correct / labels.shape[0]

class BinomialBayesClassifier():

    def __init__(self, posVocab, negVocab, neutralVocab, total_train_examples):
        self.master_vocab = {
            "positive": posVocab,
            "negative": negVocab,
            "neutral": neutralVocab
        }
        self.total = total_train_examples
        self.probability_vectors = {} #holds p_words given class after fit
        self.probabilities = {} #holds probability of sentiment after fit


    #hmmm do we need trainY here? don't think so but eh
    def fit(self, trainX, sentiment):
        if sentiment == "positive" or sentiment == "negative" or sentiment == "neutral":
            self.vocab = self.master_vocab[sentiment]
        else:
            print("Error in Fit: sentiment must be \"positive\", \"negative\", or \"neutral\"")
            return

        train_bow = self._bag_words(trainX)#only saved for access for assignment output
        self.probability_vectors[sentiment] = self._p_words_given_class(train_bow)
        self.probabilities[sentiment] = trainX.shape[0] / self.total


    def extract_phrases(self, tweet):
        #print("extracting phrases from")
        #print(tweet[1][0])

        subphrases = []
        full_phrase = tweet[1][0]
        print()
        print("Subphrases for: ", full_phrase)
        if (full_phrase != "" and not isinstance(full_phrase, float)):
            words = full_phrase.split()
            for strt, end in combinations(range(len(words)), 2):
                print(words[strt:end])
                subphrases.append(words[strt:end])

        return subphrases


    #INCOMPLETE
    #takes in np array of tweets and sentiments
    #returns predicted phrases for each tweet
    def predict_tweets(self, X):
        # For each tweet and sentiment pair, get predicted phrase
        preds = [self.predict_tweet(x) for x in X.iterrows()]
        return np.array(preds)


    def predict_tweet(self, X):
        # Extract phrases from tweet
        phrases = self.extract_phrases(X)
        return True


        # Turn each tweet into bag of words
        """
        self.test_bow = self._bag_words(X) #only saved for access for assignment output
        preds = [self._predict(x) for x in self.test_bow]
        return np.array(preds)

        """



    #INCOMPLETE
    #takes in jagged np array of strings of examples
    def predict(self, X):
        self.test_bow = self._bag_words(X) #only saved for access for assignment output
        preds = [self._predict(x) for x in self.test_bow]
        return np.array(preds)

    #INCOMPLETE
    #predict a single example
    def _predict(self, x):
        if self._p_class_given_x(x, "pos") > self._p_class_given_x(x,"neg"):
            return 1
        else:
            return 0

    #takes in a jagged np array of strings of examples and a vocabulary
    #returns a binomal bag of words (num examples rows, num words in vocab cols)
    def _bag_words(self, data):
        num_examples = data.shape[0]
        num_features = self.vocab.shape[0]
        #make zero matrix with numex rows and numft columns
        matrix = np.zeros((num_examples, num_features), dtype="int8")
        for row_ind in range(num_examples): #for each review
            for row_word in data[row_ind]:
                #match vocab indexes for each word in review, and set to 1
                matrix[row_ind][np.where(row_word==self.vocab)]=1
        return matrix

    #function for training
    #takes training features BOW (X), training labels (Y), and optional uniform dirichlet prior value (default 1)
    #also sets self.ppos and self.pneg
    #returns 2 row vectors of probability_vectors - [p(w0|y=0), p(w1|y=0)...], [p(w0|y=1), p(w1|y=1)...]
    def _p_words_given_class(self, X, alpha=1):
        numerator = np.sum(X, axis=0) +alpha#vector of word counts in features matrix + alpha
        denominator = np.sum(numerator)   #total number of words in class + |V|alpha
        return numerator / denominator


    #INCOMPLETE
    #helper function for _predict; calculates probability of example X being in class cl ("pos" or "neg")
    def _p_class_given_x(self, x, cl):
        if cl == "pos":
            return np.sum(x*np.log(self.pos_wordprobs)) + np.log(self.ppos)
        else:
            return np.sum(x*np.log(self.neg_wordprobs)) + np.log(self.pneg)


train, test = load_data()
posTrain, negTrain, posVocab, negVocab = clean_train_data(train)
posTrainX, posTrainY = posTrain
negTrainX, negTrainY = negTrain



#need to add neutral vocab, but haven't yet
classifier = BinomialBayesClassifier(posVocab, negVocab, negVocab, len(train.index) )

classifier.fit(posTrainX, "positive")
classifier.fit(negTrainX, "negative")
#call fit on neutral data too.

# Get phrases based off of tweets and associated sentiments
train_predictions = classifier.predict_tweets(train[['text', 'sentiment']])
# test_predictions = classifier.predict(testX)

# Compare predicted phrases with actual phrases
# train_accuracy = accuracy_score(train_predictions,trainY)
# test_accuracy = accuracy_score(test_predictions, testY)

# output_info(classifier.train_bow, trainY, vocabulary, "preprocessed_train.txt")
# output_info(classifier.test_bow, testY, vocabulary, "preprocessed_test.txt")

# with open("results.txt","w") as fp:
#     print("Results from ./trainingSet.txt and ./testSet.txt:",file=fp)
#     print("\tTrain Accuracy:", train_accuracy, file=fp)
#     print("\tTest Accuracy:", test_accuracy, file=fp)

# print("Results from ./trainingSet.txt and ./testSet.txt:")
# print("\tTrain Accuracy:", train_accuracy)
# print("\tTest Accuracy:", test_accuracy)





