import pandas as pd
from textblob.classifiers import NaiveBayesClassifier as NBC


#Removing noisy words from text


def remove_noise(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to"]
    words = input_text[1].split(" ")
    noise_free_words = [word for word in words if word not in dirty_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

#Text classification
def classification_text(input_text):
    training_corpus = [
        ('They are bitches and whores', 'Hate'),
        ("Stupid cunt and piece of fucking shit", 'Hate'),
        ('Bastardo and motherfucker with cock', 'Hate'),
        ('This bitch and cunt has big pussy', 'Hate'),
        ('I think that they are very trustworthy', 'Love'),
        ('I\'m loving it', 'Love'),
        ('I feel very good about these dates.', 'Love'),
        ('This is joy', 'Love'),
        ("Pure happiness", 'Love'),
        ('Donald trump fucks nazi hitler in the ass', 'Hate')]
    test_corpus = [
        ('Comparison is straightforward = you are shit', 'Hate'),
        ('I feel brilliant!', 'Love'),
        ('Gary is a friend of mine.', 'Love'),
        ("I hope you die", 'Hate'),
        ('The date was good.', 'Class_A'),
        ('I do not enjoy my job', 'Hate')]


#Load data from csv

datafile = pd.read_csv('test_tweets.csv')
df = pd.read_csv('test_tweets.csv')
df.describe()
print(df)
print("before removing noise")
df['tweet'] = df.apply(remove_noise, axis=1)
print("after removing noise")
print(df)
#model = NBC(df[1])


#Vectorization


