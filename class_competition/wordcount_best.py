import pandas as pd 
import numpy as np

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Import the string dictionary that we'll use to remove punctuation
import string

# Import datasets

#train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
#test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
#sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print("loading data ... ")
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sample_submission.csv')

# The row with index 13133 has NaN text, so remove it from the dataset

train[train['text'].isna()]

train.drop(314, inplace = True)

import re
def clean_text(x):
    #omit URLS
    #x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x)
    return x.lower()

# Make all the text lowercase - casing doesn't matter when 
# we choose our selected text.
train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Make training/test split
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(
    train, train_size = 0.80, random_state = 0)

pos_train = X_train[X_train['sentiment'] == 'positive']
neutral_train = X_train[X_train['sentiment'] == 'neutral']
neg_train = X_train[X_train['sentiment'] == 'negative']

# Use CountVectorizer to get the word counts within each dataset

#stop_words = set(['if', 'with',"\'" ,'through', 'm', 'off', 'y', 'have', 'an', 'up', 'll', 'has', 'own', 'will', 'me', "you'd", 'most', 'did', 'they', 'the', 'my', 'on', 'over', 's', 'while', 'who', "you're", 'down', 'out', 'some', 'where', 'myself', 'yourselves', 'you', 'him', 'her', 'am', 'of', 'ourselves', 'was', 'whom', 'does', 'do', 'just', 'had', 'or', 'their', 'about', 'more', 've', 'his', 'against', 'himself', 'because', 'each', 'any', 'are', 'hers', 'it', 'very', "you'll", 'he', 'i', 'what', 'that', 'above', 'ma', 'why', "that'll", 'once', 'them', 'having', 'when', 'this', 'there', 'a', 'before', 'below', 'but', 'now', 'o', 'is', 'to', 'yours', 'other', 'theirs', 'doing', 'under', 'were', 'we', 'which', 'itself', "you've", 'being', 'both', "it's", 'how', 'she', 'same', 'until', 'than', 'your', 'after', 'so', 'yourself', 'd', "should've", 'these', 'be', 'into', 'here', 'themselves', "she's", 'herself', 'as', 'should', 'by', 'too', 'then', 'all', 'its', 'such', 'during', 'for', 'in', 't', 'been', 'at', 'wasn', 'few', 're', 'those', 'and', 'ours', 'between', 'from', 'further', 'our', 'only'])

cv = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=10000,
                                    #stop_words=stop_words,
                                     #consider removing not again to see if it improves
                                    stop_words = ENGLISH_STOP_WORDS-set("not"),
                                    ngram_range=(1,2),
                                    #token_pattern=r"(?u)\b\w\w+\b|!|\?|\'",
                                    )
print("learning...")
X_train_cv = cv.fit_transform(X_train['selected_text'])
inverted_vocab = dict([v,k] for k,v in cv.vocabulary_.items())
X_pos = cv.transform(pos_train['selected_text'])
X_neutral = cv.transform(neutral_train['selected_text'])
X_neg = cv.transform(neg_train['selected_text'])

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

# Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that 
# contain those words

pos_words = {}
neutral_words = {}
neg_words = {}

#print(len([x.split() for x in cv.get_feature_names()]) == len(cv.get_feature_names()))
for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    
    pos_words[k] = pos/pos_train.shape[0]
    neutral_words[k] = neutral/neutral_train.shape[0]
    neg_words[k] = neg/neg_train.shape[0]
    
# We need to account for the fact that there will be a lot of words used in tweets of every sentiment.  
# Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other 
# sentiments that use that word.

neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    #print(tweet)
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        # Calculate the sum of weights for each word in the substring
#         for p in range(len(lst[i])):
#             if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
#                 new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
        
        bow = cv.transform(lst[i]).toarray()[0]
        inds = np.where(bow != 0)[0].flatten()
        for j in list(inds):
            if bow[j] != 0:
                word = inverted_vocab[j]
                new_sum += dict_to_use[word]
        # If the sum is greater than the score, update our current selection
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)

pd.options.mode.chained_assignment = None

tol = 0.001

X_val['predicted_selection'] = ''

print("making predictions...")
for index, row in X_val.iterrows():
    
    selected_text = calculate_selected_text(row, tol)
    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))
