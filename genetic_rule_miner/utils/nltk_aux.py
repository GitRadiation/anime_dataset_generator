import nltk


def download_nltk_resources():
    try:
        # Check if 'stopwords' and 'punkt' are downloaded
        nltk.data.find("corpora/stopwords.zip")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
