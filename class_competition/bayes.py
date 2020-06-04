#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import re
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

    review = str(review)
    #omit URLS
    review = re.sub(r'^https?:\/\/.*[\r\n]*', '', review)

    #ignore case, separate on whitespace
    review = review.lower().split()
    return np.array(review)

#takes in pandas df of test and train data
def clean_train_data(train):
    pos = train.loc[train["sentiment"] == "positive"].to_numpy() #create np matrix out of pos rows
    neg = train.loc[train["sentiment"] == "negative"].to_numpy() #create np matrix out of neg rows

    _, posTweets, posSelectedTxt, _ = pos.T
    _, negTweets, negSelectedTxt, _ = neg.T

    posVocab = np.unique(_clean_text(list(posSelectedTxt)))
    negVocab = np.unique(_clean_text(list(negSelectedTxt)))
    # posVocab = np.unique(_clean_text(list(posTweets)))
    # negVocab = np.unique(_clean_text(list(negTweets)))

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

def _jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

#expects 1D np arrays of equal dimensions:
def accuracy_score(predicted, actual):
    num_examples = actual.shape[0]
    jaccards = [_jaccard(predicted[i],actual[i]) for i in range(num_examples)]
    score = np.sum(jaccards) / num_examples
    return score

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
        # if sentiment == "positive" or sentiment == "negative" or sentiment == "neutral":
        #     self.vocab = self.master_vocab[sentiment]
        # else:
        #     print("Error in Fit: sentiment must be \"positive\", \"negative\", or \"neutral\"")
        #     return

        train_bow = self._bag_words(trainX, sentiment)#only saved for access for assignment output
        self.probability_vectors[sentiment] = self._p_words_given_class(train_bow)
        self.probabilities[sentiment] = trainX.shape[0] / self.total


    def extract_phrases(self, tweet):
        #print("extracting phrases from")
        #print(tweet[1][0])

        subphrases = []
        #full_phrase = tweet[1][0]
        full_phrase = tweet
        print()
        #print("Subphrases for: ", full_phrase)
        #if (full_phrase != "" and not isinstance(full_phrase, float)):
        #    words = full_phrase.split()
        words = full_phrase
        for strt, end in combinations(range(len(words) + 1), 2):
            #print(words[strt:end])
            subphrases.append(words[strt:end])

        return subphrases


    #INCOMPLETE
    #takes in np array of tweets and sentiments
    #returns predicted phrases for each tweet
    def predict_tweets(self, X):
        # For each tweet and sentiment pair, get predicted phrase
        preds = [self.predict_tweet(x) for x in X]
        return np.array(preds)


    def predict_tweet(self, X, sentiment):
        # Extract phrases from tweet
        #check for neutral tweet and return whole tweetnegative
        phrases = self.extract_phrases(X)
        pos_bow = self._bag_words(np.array(phrases), "positive")
        print("pos_bow:", pos_bow)
        neg_bow = self._bag_words(np.array(phrases), "negative")
        bow_pred = self._predict(pos_bow,neg_bow, sentiment)
        #turn list of words back into string
        #print("bow_pred:", bow_pred)
        pred_inds = np.where(bow_pred == 1)
        #print("pred_inds:", pred_inds)
        prediction = self.master_vocab[sentiment][pred_inds]
        #print("pred word list:", prediction)
        prediction = " ".join(prediction)
        #print("prediction:", prediction)
        return prediction


        # Turn each tweet into bag of words
        """
        self.test_bow = self._bag_words(X) #only saved for access for assignment output
        preds = [self._predict(x) for x in self.test_bow]
        return np.array(preds)

        """

    #INCOMPLETE
    #predict a single example
    #x is  bow phrases of a whole tweet
    def _predict(self, pos_bow,neg_bow, sentiment):
        probs = []
        if sentiment == "positive":
            for rowind in range(pos_bow.shape[0]):
                pphrase = pos_bow[rowind]
                nphrase = neg_bow[rowind]
                pinds = np.where(pphrase == 1)
                ninds = np.where(nphrase == 1)
                posprob = np.sum(self.probability_vectors["positive"][pinds])
                negprob =np.sum(self.probability_vectors["negative"][ninds])
                probs.append(posprob-negprob)
            #print("bow", pos_bow)
            #print("probs", probs, len(probs))
            predict_ind = np.argmax(probs)
            #print("predict phrase at row:", predict_ind)
            return pos_bow[predict_ind]

        elif sentiment == "negative":
            for rowind in range(neg_bow.shape[0]):
                pphrase = pos_bow[rowind]
                nphrase = neg_bow[rowind]
                pinds = np.where(pphrase == 1)
                ninds = np.where(nphrase == 1)
                posprob = np.sum(self.probability_vectors["positive"][pinds])
                negprob =np.sum(self.probability_vectors["negative"][ninds])
                probs.append(negprob - posprob)
            predict_ind = np.argmax(probs)
            return neg_bow[predict_ind]

        else:
            print(str(sentiment)+"must be \"positive\" or \"negative\".")
            return

    #takes in a jagged np array of strings of examples and a vocabulary
    #returns a binomal bag of words (num examples rows, num words in vocab cols)
    def _bag_words(self, data, sentiment):
        num_examples = data.shape[0]
        num_features = self.master_vocab[sentiment].shape[0]
        #make zero matrix with numex rows and numft columns
        matrix = np.zeros((num_examples, num_features), dtype="int8")
        for row_ind in range(num_examples): #for each review
            for row_word in data[row_ind]:
                #match vocab indexes for each word in review, and set to 1
                matrix[row_ind][np.where(row_word==self.master_vocab[sentiment])]=1
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

posPreds = classifier.predict_tweets(posTrainX, "positive")
negPreds = classifier.predict_tweets(negTrainX, "negative")

posscore = accuracy_score(posPreds, posTrainY)
negscore = accuracy_score(negPreds, negTrainY)

print("Total Score:", posscore+negscore)

print(posscore)
#numpy.savetxt("submission.csv", submitValues, delimeter=",")

# Get phrases based off of tweets and associated sentiments
#train_predictions = classifier.predict_tweets(train[['text', 'sentiment']])
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





