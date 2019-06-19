import pandas as pd

from util import DataTranslators
#Removing noisy words from text

#Load data from csv

datafile = pd.read_csv('test_tweets.csv')

for index, row in datafile.iterrows():
    print(row[DataTranslators.ca.name])

df.describe()
df.info()

noise_list = ["is", "a", "this", "..."]
def _remove_noise(input_text):
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

print(_remove_noise("this is a sample text"))




