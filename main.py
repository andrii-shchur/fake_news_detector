import nltk

from fastapi import FastAPI

from detector import Detector
from models import Article

app = FastAPI()


@app.on_event('startup')
async def startup():
    nltk.download('stopwords')
    nltk.download('wordnet')


@app.post('/detect')
async def detect(content: Article):
    detector = Detector(content)
    detection = detector.detect()

    return {'detection': detection}
