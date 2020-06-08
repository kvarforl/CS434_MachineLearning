from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations
from copy import *
import re
import argparse
import pandas as pd
import numpy as np

submission_for_kaggle = False

stop_words = set(['if', 'with', 'through', 'm', 'off', 'y', 'have', 'an', 'up', 'll', 'has', 'own', 'will', 'me', "you'd", 'most', 'did', 'they', 'the', 'my', 'on', 'over', 's', 'while', 'who', "you're", 'down', 'out', 'some', 'where', 'myself', 'yourselves', 'you', 'him', 'her', 'am', 'of', 'ourselves', 'was', 'whom', 'does', 'do', 'just', 'had', 'or', 'their', 'about', 'more', 've', 'his', 'against', 'himself', 'because', 'each', 'any', 'are', 'hers', 'it', 'very', "you'll", 'he', 'i', 'what', 'that', 'above', 'ma', 'why', "that'll", 'once', 'them', 'having', 'when', 'this', 'there', 'a', 'before', 'below', 'but', 'now', 'o', 'is', 'to', 'yours', 'other', 'theirs', 'doing', 'under', 'were', 'we', 'which', 'itself', "you've", 'being', 'both', "it's", 'how', 'she', 'same', 'until', 'than', 'your', 'after', 'so', 'yourself', 'd', "should've", 'these', 'be', 'into', 'here', 'themselves', "she's", 'herself', 'as', 'should', 'by', 'too', 'then', 'all', 'its', 'such', 'during', 'for', 'in', 't', 'been', 'at', 'wasn', 'few', 're', 'those', 'and', 'ours', 'between', 'from', 'further', 'our', 'only'])

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


class BinomialBayesClassifier():

    def __init__(self):

        self.vectorizer = CountVectorizer(min_df=1,
                            ngram_range=(1,2), #consider sentiments in pairs of words. can change to any range
                            preprocessor=_clean_text,
                            stop_words= stop_words)

        #vocabulary dictionary's are accessible via self.cvs[sentiment].vocabulary_
        #can be used to access index of word {word: index}
        #can also be used to create BOW representation -- use self.cvs[sentiment].transform(X)
        self.cvs = {} #a dictionary of vectorizer after fit is called on positive and negative data respectively

        self.inverted_vocabularies = {} #used to access the word at index {index: word}
        self.probability_vectors = {} #holds p_words given class after fit for each sentiment

    #call on column of selected text in pandas df
    def fit(self, trainX, sentiment):
        self.cvs[sentiment] = deepcopy(self.vectorizer.fit(trainX))
        train_bow = self.cvs[sentiment].transform(trainX) #creates bow representation (multinomial)
        self.inverted_vocabularies[sentiment] = dict([[v,k] for k,v in self.cvs[sentiment].vocabulary_.items()])
        self.probability_vectors[sentiment] = self._p_words_given_class(train_bow, sentiment)

    #TODO: this needs to be adjusted to be applied to the BOW representation of a tweet if we want to use ngrams (which I think we should :)
    def extract_phrases(self, tweet ):
        subphrases = []
        #words = [w for w in tweet if w in self.inverted_vocabularies[sentiment]]
        words = tweet.split()
        for strt, end in combinations(range(len(words) + 1), 2):
            subphrases.append(" ".join(words[strt:end]))
        return pd.DataFrame(subphrases, columns=["phrases"]).to_numpy()

    def predict_tweets(self, X, sentiment):
        # For each tweet and sentiment pair, get predicted phrase
        X = X.to_numpy()
        preds = [self.predict_tweet(x, sentiment) for x in X]
        return np.array(preds)


    def predict_tweet(self, X, sentiment):
        # Extract phrases from tweet
        #check for neutral tweet and return whole tweet
        #print("tweet in predict_tweet:",X)
        if sentiment == "neutral":
            return " ".join(X)
        if not list(X):
            return ""
        phrases = self.extract_phrases(X)
        #print("phrases:", phrases)
        #apply to each row of phrases!
        pos_bow = np.apply_along_axis(self.cvs["positive"].transform, axis=1, arr=phrases)
        pos_bow = np.asarray([x.toarray() for x in pos_bow])
        neg_bow = np.apply_along_axis(self.cvs["negative"].transform, axis=1, arr=phrases)
        neg_bow = np.asarray([x.toarray() for x in neg_bow])
        #print("pos_bow:", pos_bow, "neg_bow:", neg_bow)
        bow_pred = self._predict(pos_bow,neg_bow, sentiment)
        print(bow_pred)
        quit()
        prediction = [self.inverted_vocabularies[sentiment][i] for i in range(len(bow_pred)) if bow_pred[i] != 0] #gen list of words
        prediction = " ".join(prediction)#make into string
        return prediction

    #predict a single example
    def _predict(self, pos_bow,neg_bow, sentiment):
        probs = []
        if sentiment == "positive":
            for rowind in range(pos_bow.shape[0]):
                pphrase = pos_bow[rowind]
                nphrase = neg_bow[rowind]
                pinds = np.where(pphrase != 0)
                ninds = np.where(nphrase != 0)
                posprob = np.sum(self.probability_vectors["positive"][pinds])
                negprob =np.sum(self.probability_vectors["negative"][ninds])
                probs.append(posprob-negprob)
            predict_ind = np.argmax(probs)
            return pos_bow[predict_ind]

        elif sentiment == "negative":
            for rowind in range(neg_bow.shape[0]):
                pphrase = pos_bow[rowind]
                nphrase = neg_bow[rowind]
                pinds = np.where(pphrase != 1)
                ninds = np.where(nphrase != 1)
                posprob = np.sum(self.probability_vectors["positive"][pinds])
                negprob =np.sum(self.probability_vectors["negative"][ninds])
                probs.append(negprob - posprob)
            predict_ind = np.argmax(probs)
            return neg_bow[predict_ind]

        else:
            print(str(sentiment)+"must be \"positive\" or \"negative\".")
            return


    #function for training
    #takes training features BOW (X), sentiment, and optional uniform dirichlet prior value (default 1)
    #returns 2 row vectors of probability_vectors - [p(w0|y=0), p(w1|y=0)...], [p(w0|y=1), p(w1|y=1)...]
    def _p_words_given_class(self, X,sentiment, alpha=1):
        numerator = np.sum(X, axis=0) +alpha#vector of word counts in features matrix + alpha
        denominator = np.sum(numerator)   #total number of words in class + |V|alpha
        vector = numerator / denominator
        return vector


print("loading data..")
#these are pandas dataframes
#Here's how to access things:
#   tweets: name_of_dataframe["text"]
#   selected text: name_of_datafram["selected_text"]
#   keys: name_of_dataframe["textID"]
#   sentiment: name_of_datafram["sentiment"]
alltrain, alltest = load_data()

#remove neutral examples for log regression classifier
#these are still pandas dataframes (they work most nicely with countvectorizer
posTrain = alltrain[alltrain["sentiment"] =="positive"]
negTrain = alltrain[alltrain["sentiment"] =="negative"]
test = alltest[alltest["sentiment"] != "neutral"]

classifier = BinomialBayesClassifier()
classifier.fit(posTrain["selected_text"],"positive")
classifier.fit(negTrain["selected_text"], "negative")

posPreds = classifier.predict_tweets(posTrain["text"], "positive")
print(posPreds)


