import pandas as pd
from nltk import WordNetLemmatizer
from classification import classification
from textblob.classifiers import NaiveBayesClassifier as NBC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "multiplying"
#Removing noisy words from text


def remove_noise(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to", "ate", "something", "the", "how", "my", "at"]
    words = input_text[1].split(" ")
    noise_free_words = [word for word in words if word not in dirty_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

def remove_noise_train(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to", "ate", "something", "the", "how", "my", "at"]
    words = input_text[2].split(" ")
    noise_free_words = [word for word in words if word not in dirty_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

#Text classification
classification.text()


#Load data from csv

df = pd.read_csv('test_tweets.csv')
traindf = pd.read_csv('train_test_tweets.csv')
df.describe()
print(df)
#Lemmatize
df['tweet'] = df.apply(remove_noise, axis=1)
traindf['tweet'] = traindf.apply(remove_noise_train, axis=1)
print("after removing noise")
print(df)
#model = NBC(df[1])


#Vectorization
tfdid = TfidfVectorizer(max_features = 50000, lowercase=True, analyzer='word',
                                 stop_words='english',ngram_range=(1,1))
train_vect = tfdid.fit_transform((df['tweet']))

#Classification
classification_text()

