import pandas as pd
from nltk import WordNetLemmatizer
import classification
from textblob.classifiers import NaiveBayesClassifier as NBC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
lem = WordNetLemmatizer()


class classification():
    def classification_text(self, ):
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