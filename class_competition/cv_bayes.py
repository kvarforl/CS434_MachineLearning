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

    def __init__(self):

        self.vectorizer = CountVectorizer(min_df=1,
                            ngram_range=(1,2), #consider sentiments in pairs of words. can change to any range
                            preprocessor=_clean_text,
                            #token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\*|\.|,|:|;",
                            token_pattern=r'[^\s]+',
                            stop_words= stop_words
                            )

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

    def addRegexSyntax(self, word):
        #return "(" + re.sub("[\W_]", " ", word) + ")"
        return "(" + re.escape(word) + ")"

    # Takes a tweet and a list of words,
    # finds the full text substring that the word list was made from
    def reconstructText(self, tweet, words):
        if not words or tweet == "":
            return ""

        anyContainedWord = "(" + "|".join(list(map(self.addRegexSyntax, words))) + ")"
        regexLine = "(" + anyContainedWord + "(.*)){" + str(len(words) - 1) + "}" + anyContainedWord

        #match = re.search(regexLine, re.sub("[\W_]", " ", tweet.lower()))
        match = re.search(regexLine, tweet.lower())

        if (match == None):
            print("ERROR: No possible substring found when reconstructing tweet line. This should not happen, something is wrong...")
            print("Tweet Text: ")
            print(tweet)
            print("searching for words: ", words)
            quit()

        extractedText = tweet[match.start():match.end()]
        return extractedText


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
            return X
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
        bow_pred = self._predict(pos_bow,neg_bow, sentiment)[0]
        #print(bow_pred)
        prediction = [self.inverted_vocabularies[sentiment][i] for i in range(len(bow_pred)) if bow_pred[i] != 0] #gen list of words
        prediction = " ".join(prediction)#make into string
        prediction = set(prediction.split())


        # Aiden's regex text extraction experiment :)
        return self.reconstructText(X, list(prediction))


        #prediction = " ".join(prediction)#make into string
        #return prediction

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
                pinds = np.where(pphrase != 0)
                ninds = np.where(nphrase != 0)
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
neutralTrain = alltrain[alltrain["sentiment"] =="neutral"]

posTest = alltest[alltest["sentiment"] =="positive"]
negTest = alltest[alltest["sentiment"] =="negative"]
neutralTest = alltest[alltest["sentiment"] =="neutral"]

classifier = BinomialBayesClassifier()
classifier.fit(posTrain["selected_text"],"positive")
classifier.fit(negTrain["selected_text"], "negative")

if submission_for_kaggle:
    posPreds = classifier.predict_tweets(posTest["text"], "positive")
    negPreds = classifier.predict_tweets(negTest["text"], "negative")
    neutralPreds = classifier.predict_tweets(neutralTest["text"], "neutral")

    print("Building submission file...")
    posSubmit = np.column_stack((posTest["textID"].to_numpy(), posPreds))
    negSubmit = np.column_stack((negTest["textID"].to_numpy(), negPreds))
    neutralSubmit = np.column_stack((neutralTest["textID"].to_numpy(), neutralPreds))
    labels = np.array([["textID", "selected_text"]])
    submissionMatrix = np.concatenate((labels, posSubmit, negSubmit, neutralSubmit))
    np.savetxt("submission.csv", submissionMatrix, delimiter=",", fmt='"%s"')

else:
    posPreds = classifier.predict_tweets(posTrain["text"], "positive")
    negPreds = classifier.predict_tweets(negTrain["text"], "negative")
    neutralPreds = classifier.predict_tweets(neutralTrain["text"], "neutral")

    print("building ouput file...")
    """
    posSubmit = np.column_stack((posTrain["text"].to_numpy(), posPreds, posTrain["selected_text"].to_numpy()))
    negSubmit = np.column_stack((negTrain["text"].to_numpy(), negPreds, negTrain["selected_text"].to_numpy()))
    neutralSubmit = np.column_stack((neutralTrain["textID"].to_numpy(), neutralPreds, neutralTrain["selected_text"].to_numpy()))
    labels = np.array([["textID", "predicted", "actual"]])
    submissionMatrix = np.concatenate((labels, posSubmit, negSubmit, neutralSubmit))
    np.savetxt("cv_bayes_train.csv", submissionMatrix, delimiter=",", fmt='"%s"')
    """
    posSubmit = np.column_stack((posTrain["textID"].to_numpy(), posPreds, posTrain["selected_text"].to_numpy(), posTrain["sentiment"].to_numpy()))
    negSubmit = np.column_stack((negTrain["textID"].to_numpy(), negPreds, negTrain["selected_text"].to_numpy(), negTrain["sentiment"].to_numpy()))
    neutralSubmit = np.column_stack((neutralTrain["textID"].to_numpy(), neutralPreds, neutralTrain["selected_text"].to_numpy(), neutralTrain["sentiment"].to_numpy()))
    labels = np.array([["textID", "selected_text", "actual_text", "sentiment"]])
    submissionMatrix = np.concatenate((labels, posSubmit, negSubmit, neutralSubmit))
    np.savetxt("submission.csv", submissionMatrix, delimiter=",", fmt='"%s"')



    posscore = accuracy_score(posPreds, posTrain["selected_text"].to_numpy())
    negscore = accuracy_score(negPreds, negTrain["selected_text"].to_numpy())
    neutralscore = accuracy_score(neutralPreds, neutralTrain["selected_text"].to_numpy())
    print("Total Score:", (posscore+negscore+neutralscore)/3)
    print("neg:", negscore, "pos:", posscore, "neut:", neutralscore)


