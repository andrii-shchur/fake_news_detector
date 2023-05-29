# Fake News Detector

Fake News Detector helps you determine whether the article is true or you are being lied to

## Installation
Fake News Detector requires [Python 3.11](https://www.python.org/downloads/release/python-3113/) to run

Clone this repository
```
git clone https://github.com/andrii-shchur/fake_news_detector.git
```

Install `pipenv`
```
python -m pip install pipenv
```

Install dependencies from `Pipfile`
```
pipenv install
```

# Run application
Run the server
```
uvicorn main:app
```

Great! Now backend server is running on `http://localhost:8000`

Let's debunk fake news using GUI! Open `index.html` using your favorite web-browser, and you're good to go💪