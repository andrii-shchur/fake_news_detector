import pickle
import re

import joblib
import numpy as np

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.train import Checkpoint


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from models import Article



# self._model = pickle.load()

# self._model = load_model('changed_dropout')
# self._model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
class Detector:
    def __init__(self, content: Article):
        self._content = content
        self.one_hot_n = 10000
        self.pad_length = 200

        # We tried different ways to save/load model to use it in this web app
        # Method 1: by loading model from model saved using 'model.save("path")'
        self._model = load_model('model')
        # Method 2: by loading model from h5 weights file
        self._model = load_model('model.h5')
        # Method 3: by loading json config and restoring model from checkpoint
        with open('model/model.json', 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        checkpoint = Checkpoint(model=model)
        checkpoint.restore("model/model.ch-1")
        self._model = model
        # Method 4: by loading model using pickle module
        # Note: methods below work properly only for Linux
        # because they use tempfile module that doesn't work properly on Windows
        with open('model.pickle', 'rb') as f:
            self._model = pickle.load(f)
        # Method 5: by unpickling using joblib
        self._model = joblib.load('model.pkl')

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
