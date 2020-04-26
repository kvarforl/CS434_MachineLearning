import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def make_plot(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="Alpha (a)",
            ylabel = "Accuracy",
            title=title)
    plt.xticks(np.arange(0, 2.2, 0.2));
    fig.savefig(title.replace(" ", "_")+".png")
    plt.show()

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
test_features = features[40000: , :]

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

# Find useful numbers
labelArray = imdb_labels.to_numpy()
train_labels = labelArray[0:30000, :]
validation_labels = labelArray[30000: , :]

p_positive_training = np.count_nonzero(train_labels == "positive") / len(train_labels)
p_negative_training = 1 - p_positive_training
p_positive_validation = np.count_nonzero(validation_labels == "positive") / len(validation_labels)
p_negative_validation = 1 - p_positive_validation

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

def train_vector(pos_features, neg_features, alpha):
    wpositive = p_wi_given_y(pos_features, alpha)
    wnegative = p_wi_given_y(neg_features, alpha)
    return (wpositive, wnegative)

def get_predictions(p_probs, n_probs):
    predictions = []
    for i in range(len(p_probs)):
        if (p_probs[i] > n_probs[i]):
            predictions.append("positive")
        else:
            predictions.append("negative")
    return predictions

def get_accuracy(wpositive, wnegative):

    def map_positive_prob(feature):
        return calc_p_y_given_x(feature, wpositive, p_positive_training)

    def map_negative_prob(feature):
        return calc_p_y_given_x(feature, wnegative, p_negative_training)

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

    return overall_accuracy

trained_vectors = []
vector_accuracies = []
alpha_values = []

highest_accuracy = 0
best_vector= 0

for i in range(0, 11):
    a = float(i) * 0.2
    alpha_values.append(a)
    trained_vectors.append(train_vector(positive_features_training, negative_features_training, a))
    vector_accuracies.append(get_accuracy(trained_vectors[-1][0], trained_vectors[-1][1]))
    #print(vector_accuracies[-1])

    # keep track of best vector
    if (highest_accuracy <= vector_accuracies[-1]):
        highest_accuracy = vector_accuracies[-1]
        best_vector = trained_vectors[-1]

# Plot accuracy vs alpha values
make_plot(alpha_values, vector_accuracies, "validation accuracy vs. alpha")

def generate_test_predictions(positive_v, negative_v):
    def map_positive_prob(feature):
        return calc_p_y_given_x(feature, positive_v, p_positive_training)

    def map_negative_prob(feature):
        return calc_p_y_given_x(feature, negative_v, p_negative_training)

    positive_probs = list(map(map_positive_prob, test_features))
    negative_probs = list(map(map_negative_prob, test_features))
    test_predictions = get_predictions(positive_probs, negative_probs)
    return test_predictions

# Generate and output predictions using best vector on testing data
test_predictions = generate_test_predictions(best_vector[0], best_vector[1])
with open("test-prediction2.csv", "w") as fp:
    for c in test_predictions:
        if(c == "positive"):
            print("1", file=fp)
        else:
            print("0",file=fp)
