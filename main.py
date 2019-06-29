import string

import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, model_selection, metrics, preprocessing, linear_model, ensemble

lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

#Removing noisy words from text
def remove_noise(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to", "ate", "something", "the", "how", "my", "at"]
    words = input_text[1].split(" ")
    noise_free_words = [lem.lemmatize(word, 'v') for word in words if word not in dirty_words]
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
traindf.dropna(inplace=True)
df.describe()
print("Data before lemmatizing")
print(df)
#Lemmatize
df['tweet'] = df.apply(remove_noise, axis=1)
traindf['tweet'] = traindf.apply(remove_noise_train, axis=1)
print("after removing noise")
print(df)

##Not working here
#model = NBC(df[1])


#Validation set
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(traindf['id'], traindf['tweet'])

#Vectorization
tfdid = TfidfVectorizer(max_features = 5000, lowercase=True, analyzer='word',
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
        ('Donald trump fucks nazi hitler in the ass', 'Hate'),
        ('She is fucking nuts', 'Hate'),
        ('I really hate this fuckwad', 'Hate'),
        ('These puppies are beloved', 'Love')]
    test_corpus = [
        ('Comparison is straightforward = you are shit', 'Hate'),
        ('I really hate this fuckwad', 'Hate'),
        ('I feel brilliant!', 'Love'),
        ('I know this guy and I really like it', 'Love'),
        ('Gary is a friend of mine.', 'Love'),
        ('Fuck you', 'Hate'),
        ("I hope you die", 'Hate'),
        ('The date was good.', 'Love'),
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
    #TF-IDF
    prediction = model.predict(test_vectors)
    print(prediction)



#Preprocessing
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#Count vector

count_vect = CountVectorizer(analyzer = 'word', token_pattern=r'\w{1,}')
count_vect.fit(traindf['tweet'])

xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)


def train_model_bayes(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)

    #Predict labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

traindf['word_count'] = traindf['tweet'].apply(lambda x: len(x.split()))
traindf['punctuation_count'] = traindf['tweet'].apply(lambda x: len("".join(_ for _ in x if _ is string.punctuation)))


#Two classificators
accuracy = train_model_bayes(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print('accuracy', accuracy)
accuracy_random_forest = train_model_bayes(ensemble.RandomForestClassifier(), train_vect, train_y, xvalid_count)
print('classificator random forest', accuracy_random_forest)



