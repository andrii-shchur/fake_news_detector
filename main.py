import nltk

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from detector import Detector
from models import Article

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.on_event('startup')
async def startup():
    nltk.download('stopwords')
    nltk.download('wordnet')


@app.post('/detect')
async def detect(content: Article):
    detector = Detector(content)
    detection = detector.detect()

    return {'detection': detection}
