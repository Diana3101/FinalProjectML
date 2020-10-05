import pandas as pd

import re
import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


# concatenated train+val
class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset = self.dataset.drop(["Unnamed: 0"], axis=1)

        self.dataset['Processed_Reviews'] = self.dataset.review.apply(lambda x: clean_text(x))
        self.dataset.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

        return self.dataset



