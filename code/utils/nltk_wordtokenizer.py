from nltk import word_tokenize
from string import punctuation


def tokenizer(text):
    text = text.strip( )
    return [y for y in word_tokenize(text) if y not in punctuation]