#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import re
from string import punctuation
from itertools import combinations

# Change to true if this is a kaggle notebook
submission_for_kaggle = False

stop_words = set(['if', 'with', 'through', 'm', 'off', 'y', 'have', 'an', 'up', 'll', 'has', 'own', 'will', 'me', "you'd", 'most', 'did', 'they', 'the', 'my', 'on', 'over', 's', 'while', 'who', "you're", 'down', 'out', 'some', 'where', 'myself', 'yourselves', 'you', 'him', 'her', 'am', 'of', 'ourselves', 'was', 'whom', 'does', 'do', 'just', 'had', 'or', 'their', 'about', 'more', 've', 'his', 'against', 'himself', 'because', 'each', 'any', 'are', 'hers', 'it', 'very', "you'll", 'he', 'i', 'what', 'that', 'above', 'ma', 'why', "that'll", 'once', 'them', 'having', 'when', 'this', 'there', 'a', 'before', 'below', 'but', 'now', 'o', 'is', 'to', 'yours', 'other', 'theirs', 'doing', 'under', 'were', 'we', 'which', 'itself', "you've", 'being', 'both', "it's", 'how', 'she', 'same', 'until', 'than', 'your', 'after', 'so', 'yourself', 'd', "should've", 'these', 'be', 'into', 'here', 'themselves', "she's", 'herself', 'as', 'should', 'by', 'too', 'then', 'all', 'its', 'such', 'during', 'for', 'in', 't', 'been', 'at', 'wasn', 'few', 're', 'those', 'and', 'ours', 'between', 'from', 'further', 'our', 'only'])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="a .csv file of training data")
    parser.add_argument("test", help="a .csv file of test data")
    return parser.parse_args()

#returns data as pandas dataframes: use test.head() to easily inspect
def load_data():
    if (submission_for_kaggle):
        # These next two lines are specifically for the kaggle notebook
        train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv",sep="," )
        test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv",sep=",")
    else:
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
    #ignore stopwords that are not negations
    #review = [w for w in review if not w in stop_words]
    return np.array(review)

#takes in pandas df of test and train data
def clean_train_data(train):
    pos = train.loc[train["sentiment"] == "positive"].to_numpy() #create np matrix out of pos rows
    neg = train.loc[train["sentiment"] == "negative"].to_numpy() #create np matrix out of neg rows
    neutral = train.loc[train["sentiment"] == "neutral"].to_numpy()
    posKeys, posTweets, posSelectedTxt, _ = pos.T
    negKeys, negTweets, negSelectedTxt, _ = neg.T
    neutralKeys, neutralTweets, neutralSelectedTxt, _ = neutral.T

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

    cT = [_clean_text(x) for x in neutralTweets]
    cS = [_clean_text(x) for x in neutralSelectedTxt]
    neutralTweets = np.array(cT)
    neutralSelectedTxt = np.array(cS)
    #all text fields are now a jagged array of cleaned examples
    return (posTweets, posSelectedTxt), (negTweets, negSelectedTxt), (neutralTweets, neutralSelectedTxt), posVocab, negVocab, posKeys, negKeys, neutralKeys

#takes in pandas df of test data
def clean_test_data(test):
    pos = test.loc[test["sentiment"] == "positive"].to_numpy() #create np matrix out of pos rows
    neg = test.loc[test["sentiment"] == "negative"].to_numpy() #create np matrix out of neg rows
    neutral = test.loc[test["sentiment"] == "neutral"].to_numpy()
    posKeys, posTweets, _ = pos.T
    negKeys, negTweets, _ = neg.T
    neutralKeys, neutralTweets, _ = neutral.T

    # posVocab = np.unique(_clean_text(list(posTweets)))
    # negVocab = np.unique(_clean_text(list(negTweets)))

    cT = [_clean_text(x) for x in posTweets]
    posTweets = np.array(cT)

    cT = [_clean_text(x) for x in negTweets]
    negTweets = np.array(cT)

    cT = [_clean_text(x) for x in neutralTweets]
    neutralTweets = np.array(cT)

    #all text fields are now a jagged array of cleaned examples
    return (posTweets), (negTweets), (neutralTweets), posKeys, negKeys, neutralKeys


def _jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

#expects 1D np arrays of equal dimensions:
def accuracy_score(predicted, actual):
    num_examples = actual.shape[0]
    jaccards = [_jaccard(predicted[i],actual[i]) for i in range(num_examples)]
    score = np.sum(jaccards) / num_examples
    return score

class BinomialBayesClassifier():

    def __init__(self, posVocab, negVocab, total_train_examples):
        self.master_vocab = {
            "positive": posVocab,
            "negative": negVocab,
        }
        self.total = total_train_examples
        self.probability_vectors = {} #holds p_words given class after fit
        self.probabilities = {} #holds probability of sentiment after fit

    def fit(self, trainX, sentiment):
        train_bow = self._bag_words(trainX, sentiment)
        self.probability_vectors[sentiment] = self._p_words_given_class(train_bow, sentiment)
        self.probabilities[sentiment] = trainX.shape[0] / self.total


    def extract_phrases(self, tweet):
        subphrases = []
        words = tweet
        for strt, end in combinations(range(len(words) + 1), 2):
            subphrases.append(words[strt:end])

        return subphrases

    def predict_tweets(self, X, sentiment):
        # For each tweet and sentiment pair, get predicted phrase
        preds = [self.predict_tweet(x, sentiment) for x in X]
        return np.array(preds)


    def predict_tweet(self, X, sentiment):
        # Extract phrases from tweet
        #check for neutral tweet and return whole tweet
        if sentiment == "neutral":
            return " ".join(X)
        if not list(X):
            return ""
        phrases = self.extract_phrases(X)
        pos_bow = self._bag_words(np.array(phrases), "positive")
        neg_bow = self._bag_words(np.array(phrases), "negative")
        bow_pred = self._predict(pos_bow,neg_bow, sentiment)
        pred_inds = np.where(bow_pred == 1)
        prediction = self.master_vocab[sentiment][pred_inds]
        prediction = " ".join(prediction)
        return prediction

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
            predict_ind = np.argmax(probs)
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
    def _p_words_given_class(self, X,sentiment, alpha=1):
        numerator = np.sum(X, axis=0) +alpha#vector of word counts in features matrix + alpha
        denominator = np.sum(numerator)   #total number of words in class + |V|alpha
        vector = numerator / denominator
        #remove probabilities of stopwords
        inds = np.where(self.master_vocab[sentiment] == stop_words)
        vector[inds] = 0
        return vector


    #INCOMPLETE/ we never use this... but i understand why. leaving it here in case it strike inspiration
    #helper function for _predict; calculates probability of example X being in class cl ("pos" or "neg")
    # def _p_class_given_x(self, x, cl):
    #     if cl == "pos":
    #         return np.sum(x*np.log(self.pos_wordprobs)) + np.log(self.ppos)
    #     else:
    #         return np.sum(x*np.log(self.neg_wordprobs)) + np.log(self.pneg)

print("Processing training data...")
train, test = load_data()
posTrain, negTrain, neutralTrain, posVocab, negVocab, posKeys, negKeys, neutralKeys = clean_train_data(train)
posTrainX, posTrainY = posTrain
negTrainX, negTrainY = negTrain
neutralTrainX, neutralTrainY = neutralTrain


print("Training classifier...")
classifier = BinomialBayesClassifier(posVocab, negVocab, len(train.index) )

classifier.fit(posTrainX, "positive")
classifier.fit(negTrainX, "negative")

if (submission_for_kaggle):
    print("Processing testing data...")
    posTest, negTest, neutralTest, posKeys, negKeys, neutralKeys = clean_test_data(test)
    posTestX = posTest
    negTestX = negTest
    neutralTestX = neutralTest

    print("Making predictions...")
    posPreds = classifier.predict_tweets(posTestX, "positive")
    negPreds = classifier.predict_tweets(negTestX, "negative")
    neutralPreds = classifier.predict_tweets(neutralTestX, "neutral")

    print("Building submission file...")
    posSubmit = np.column_stack((posKeys, posPreds))
    negSubmit = np.column_stack((negKeys, negPreds))
    neutralSubmit = np.column_stack((neutralKeys, neutralPreds))
    labels = np.array([["textID", "selected_text"]])
    submissionMatrix = np.concatenate((labels, posSubmit, negSubmit, neutralSubmit))
    np.savetxt("submission.csv", submissionMatrix, delimiter=",", fmt='"%s"')
else:
    posPreds = classifier.predict_tweets(posTrainX, "positive")
    negPreds = classifier.predict_tweets(negTrainX, "negative")
    neutralPreds = classifier.predict_tweets(neutralTrainX, "neutral")

    posscore = accuracy_score(posPreds, posTrainY)
    negscore = accuracy_score(negPreds, negTrainY)
    neutralscore = accuracy_score(neutralPreds, neutralTrainY)
    print("Total Score:", (posscore+negscore+neutralscore)/3)
    print("neg:", negscore, "pos:", posscore)

