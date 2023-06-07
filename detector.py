import re
import numpy as np

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from models import Article


class Detector:
    def __init__(self, content: Article):
        self._content = content
        self.one_hot_n = 10000
        self.pad_length = 200
        self._model = load_model('model')
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _text_to_input(self, text: str):
        """
        Converts article contents to format suitable for model prediction

        :param text: article contents
        :return: text converted to format suitable for model prediction
        """
        text = text.lower()
        text = re.sub('[^\s\w]|[\d]', '', text)
        text = ' '.join(word for word in text.split() if word not in self.stopwords)
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())

        one_hot_text = one_hot(text, self.one_hot_n)
        res = pad_sequences([one_hot_text], padding='pre', maxlen=self.pad_length)
        return np.array(res)

    def detect(self) -> float:
        article_text = self._content.title + ' ' + self._content.text
        X = self._text_to_input(article_text)
        prediction = self._model.predict(X)

        return float(prediction[0][0])
