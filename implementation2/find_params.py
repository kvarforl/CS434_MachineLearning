import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
from scipy import stats

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

# Importing the labels
imdb_labels = pd.read_csv('IMDB_labels.csv', delimiter=',')

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
    preprocessor=clean_text,
    #max_features=2000,
)

# fit the vectorizer on the text
features = vectorizer.fit_transform(imdb_data['review'])
#features = features.toarray()
# Split feature vectors into testing and validation
train_features = features[0:30000, :]
validation_features = features[30000:40000 , :]
test_features = features[40000: , :]

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

print("percent nonzero:", features.count_nonzero() / features.getnnz())
print("num_features:", features.shape)
features.eliminate_zeros()
print("max:",features.max(), "min:",features.min(), "mean:", features.mean())

