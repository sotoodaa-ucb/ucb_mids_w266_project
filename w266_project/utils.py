import nltk


def download_nltk_resources():
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
