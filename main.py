import pandas as pd
from nltk import WordNetLemmatizer
from classification import classification

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
print(train_vect)

#Classification
def classification_text(traindf, df):
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
    train_data = []
    train_labels = []
    for row in training_corpus:
        train_data.append(row[0])
        train_labels.append(row[1])

    test_data = []
    test_labels = []
    for row in test_corpus:
        test_data.append(row[0])
        test_labels.append(row[1])

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
    # Train the feature vectors
    train_vectors = vectorizer.fit_transform(traindf)
    # Apply model on test data
    test_vectors = vectorizer.transform(df)

    # Perform classification with SVM, kernel=linear
    model = svm.SVC(kernel='linear')
    model.fit(train_vectors, train_labels)
    prediction = model.predict(test_vectors)
    print(prediction)

