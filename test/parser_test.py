
from nltk.tokenize import sent_tokenize # Sentence Tokenizer
from nltk.tokenize import word_tokenize # Word Tokenizer

import spacy
from spacy.tokenizer import Tokenizer
#from nltk.stem import PorterStemmer

from app.parser import tokenize

MY_PREAMBLE = "Friends, Romans, countrymen, lend me your ears; 911"
MY_MESSAGE = " Oh HeY there - so whatr'u  up to later???? \n   Statue of Liberty trip later. \n Text me (123) 456-7890. k cool! "

def test_custom_tokenizer():

    assert tokenize(MY_PREAMBLE) == [
        "friends", "romans", "countrymen",
        "lend", "me", "your", "ears", "911"
    ]

    assert tokenize(MY_MESSAGE) == [
        'oh', 'hey', 'there', 'so', 'whatru', 'up', 'to', 'later',
        'statue', 'of', 'liberty', 'trip', 'later',
        'text', 'me', '123', '4567890', 'k', 'cool'
    ]

    assert tokenize("Don't do Full-Time") == ["dont", "do", "fulltime"]

def test_nltk_tokenizers():

    assert sent_tokenize(MY_PREAMBLE) == [
        'Friends, Romans, countrymen, lend me your ears; 911'
    ]
    assert sent_tokenize(MY_MESSAGE) == [
        " Oh HeY there - so whatr'u  up to later????",
        'Statue of Liberty trip later.',
        'Text me (123) 456-7890. k cool!'
    ]

    assert word_tokenize(MY_PREAMBLE) == [
        'Friends', ',', 'Romans', ',', 'countrymen', ',',
        'lend', 'me', 'your', 'ears', ';', '911'
    ]
    assert word_tokenize(MY_MESSAGE) == [
        'Oh', 'HeY', 'there', '-', 'so', 'whatr', "'", 'u', 'up', 'to', 'later', '?', '?', '?', '?',
        'Statue', 'of', 'Liberty', 'trip', 'later', '.',
        'Text', 'me', '(', '123', ')', '456-7890.', 'k', 'cool', '!'
    ]

def test_spacy_tokenizer():
    nlp = spacy.load("en_core_web_md") # TODO: consider making this a text fixture
    tokenizer = Tokenizer(nlp.vocab)

    tokens = [token.text for token in tokenizer(MY_PREAMBLE)]
    assert tokens == ['Friends,', 'Romans,', 'countrymen,', 'lend', 'me', 'your', 'ears;', '911']

    tokens = [token.text for token in tokenizer(MY_MESSAGE)]
    assert tokens == [
        ' ', 'Oh', 'HeY', 'there', '-', 'so', "whatr'u", ' ', 'up', 'to', 'later????', '\n   ',
        'Statue', 'of', 'Liberty', 'trip', 'later.', '\n ',
        'Text', 'me', '(123)', '456-7890.', 'k', 'cool!'
    ]




#from sklearn.feature_extraction.text import CountVectorizer
#
#def test_sklearn_counter():
#    my_dist = FreqDist({'the': 3, 'dog': 2, 'not': 1})
#    print(my_dist.most_common(2))



#from nltk.probability import FreqDist
#
#def test_nltk_counter():
#    my_dist = FreqDist({'the': 3, 'dog': 2, 'not': 1})
#    print(my_dist.most_common(2))
