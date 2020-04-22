import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

# Importing the labels
imdb_labels = pd.read_csv('IMDB_labels.csv', delimiter=',')

# Split labels into testing and validation
labelArray = imdb_labels.to_numpy()
train_labels = labelArray[0:30000, :]
validation_labels = labelArray[30000: , :]
#print(len(train_labels))
#print(len(validation_labels))
#print(validation_labels)

def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

#takes a matrix of class y reviews, where each review is a row, and each column is a count of word occurances
#takes an alpha value for smoothing; defaults to 1
def p_wi_given_y(features, alpha=1):
    numerator = np.sum(features, axis=0) + alpha #vector of word counts in features matrix + alpha
    denominator = np.sum(features) + (features.shape[1] * alpha)  #total number of words in class + |V|alpha
    return (1/denominator) * numerator

# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000,
)

# fit the vectorizer on the text
features = vectorizer.fit_transform(imdb_data['review'])

# Split feature vectors into testing and validation
train_features = features[0:30000, :]
validation_features = features[30000:40000 , :]

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]



