import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

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


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text
)

# fit the vectorizer on the text
features = vectorizer.fit_transform(imdb_data['review'])

# Split feature vectors into testing and validation
train_features = labelArray[0:30000, :]
validation_features = labelArray[30000:40000 , :]

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

# Apply settings for part 1
vectorizer.max_features = 2000

#def is_positive_review(review)

# Find useful numbers
p_positive_training = np.count_nonzero(train_labels == "positive")
p_negative_training = np.count_nonzero(train_labels == "negative")
p_positive_validation = np.count_nonzero(validation_labels == "positive")
p_negative_validation = np.count_nonzero(validation_labels == "negative")
print("training positives: ", p_positive_training)
print("training negatives: ", p_negative_training)



