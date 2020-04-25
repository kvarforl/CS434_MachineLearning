import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

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

#takes a matrix of class y reviews, where each review is a row, and each column is a count of word occurances per review
#takes an alpha value for smoothing; defaults to 1
# returns a row vector of probabilities for each word in the vocabulary [p(w0|y), p(w1|y), ... p(wd|y)]
def p_wi_given_y(features, alpha=1):
    numerator = np.sum(features, axis=0) +alpha#vector of word counts in features matrix + alpha
    denominator = np.sum(numerator)   #total number of words in class + |V|alpha
    return (1/denominator) * numerator

#only focused on numerator for now
# x is review to be classified.
# w is the result of p_wi_givenY(positive_features) or p_wi_givenY(negative_features)
# py is either p_positive or p_negative
def calc_p_y_given_x(x, w, py):
    #no log
    #numerator = np.prod(np.power(w, x))*py
    #numerator = np.log(np.prod(np.power(w, x))) + np.log(py)

    #with log
    #numerator = np.sum(x*np.log(w.astype("float64"))) + py
    numerator = np.sum(x*np.log(w.astype("float64"))) + np.log(py)

    return numerator


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000,
)

# fit the vectorizer on the text
features = vectorizer.fit_transform(imdb_data['review'])
features = features.toarray()
# Split feature vectors into testing and validation
train_features = features[0:30000, :]
validation_features = features[30000:40000 , :]

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

# Find useful numbers
labelArray = imdb_labels.to_numpy()
train_labels = labelArray[0:30000, :]
validation_labels = labelArray[30000: , :]

p_positive_training = np.count_nonzero(train_labels == "positive") / len(train_labels)
p_negative_training = 1 - p_positive_training
#p_negative_training = np.count_nonzero(train_labels == "negative") / len(train_labels)
p_positive_validation = np.count_nonzero(validation_labels == "positive") / len(validation_labels)
p_negative_validation = 1 - p_positive_validation
#p_negative_validation = np.count_nonzero(validation_labels == "negative") / len(validation_labels)
print("probablility of training positives: ", p_positive_training)
print("probability of training negatives: ", p_negative_training)



def is_positive_review(review):
    if (review[0] == "positive"):
        return True
    else:
        return False

def is_negative_review(review):
    if (review[0] == "negative"):
        return True
    else:
        return False

# Add label column for sorting
#labeled_features_training = np.hstack((train_labels, train_features))
labeled_features_all = imdb_labels.join(pd.DataFrame(features)).to_numpy()
labeled_features_training = labeled_features_all[0:30000, :]
labeled_features_validation = labeled_features_all[30000: , :]

# Filter arrays to lists of positive and negative reviews
positive_features_training = np.array(list(filter(is_positive_review, labeled_features_training)))
negative_features_training = np.array(list(filter(is_negative_review, labeled_features_training)))
positive_features_validation = np.array(list(filter(is_positive_review, labeled_features_validation)))
negative_features_validation = np.array(list(filter(is_negative_review, labeled_features_validation)))

# Cut off label column
positive_features_training = positive_features_training[:,1:]
negative_features_training = negative_features_training[:,1:]
positive_features_validation = positive_features_validation[:,1:]
negative_features_validation = negative_features_validation[:,1:]


wpositive = p_wi_given_y(positive_features_training)
wnegative = p_wi_given_y(negative_features_training)

#WITH NO LOG
# numerator = calc_p_y_given_x(positive_features_training[0], wpositive, p_positive_training)
# denominator = numerator + calc_p_y_given_x(positive_features_training[0], wnegative, p_negative_training)
# probpostive = numerator / denominator
# print(probpostive)

# numerator = calc_p_y_given_x(positive_features_training[0], wnegative, p_negative_training)
# denominator = numerator + calc_p_y_given_x(positive_features_training[0], wpositive, p_positive_training)
# probnegative = numerator / denominator
# print(probnegative)
# print(probpostive + probnegative)

#with log
# p_positive_training = np.log(p_positive_training)
# p_negative_training = np.log(p_negative_training)
pos = calc_p_y_given_x(positive_features_training[0], wpositive, p_positive_training)
neg = calc_p_y_given_x(positive_features_training[0], wnegative, p_negative_training)
# denominator = numerator * calc_p_y_given_x(positive_features_training[0], wnegative, p_negative_training)
# probpostive = numerator - denominator
print(pos)
print(neg)
if(pos > neg):
    print("YAY?? ")
else:
    print(":((")

#print(np.log(0.5))


def map_positive_prob(feature):
    return calc_p_y_given_x(feature, wpositive, p_positive_training)

def map_negative_prob(feature):
    return calc_p_y_given_x(feature, wnegative, p_negative_training)

def get_predictions(p_probs, n_probs):
    predictions = []
    for i in range(len(p_probs)):
        if (p_probs[i] > n_probs[i]):
            predictions.append("positive")
        else:
            predictions.append("negative")
    return predictions

def calc_accuracy(predictions, goal_class):
    correct_estimations = 0
    for i in range(len(predictions)):
        if (predictions[i] == goal_class):
            correct_estimations += 1
    return correct_estimations / len(predictions)

# Generate predictions from validation data
p_positive_probs = list(map(map_positive_prob, positive_features_validation))
p_negative_probs = list(map(map_negative_prob, positive_features_validation))
p_predictions = get_predictions(p_positive_probs, p_negative_probs)
p_accuracy = calc_accuracy(p_predictions, "positive") # Accuracy of predicting positive reviews

n_positive_probs = list(map(map_positive_prob, negative_features_validation))
n_negative_probs = list(map(map_negative_prob, negative_features_validation))
n_predictions = get_predictions(n_positive_probs, n_negative_probs)
n_accuracy = calc_accuracy(n_predictions, "negative") # Accuracy of predicting negative reviews

overall_accuracy = ((p_accuracy * len(p_predictions)) + (n_accuracy * len(n_predictions))) / 10000

print("Overall Accuracy: ", overall_accuracy)


# Get accuracy by comparing with validation data labels
