

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


from nltk.tokenize import sent_tokenize # Sentence Tokenizer
from nltk.tokenize import word_tokenize # Word Tokenizer

def test_nltk_tokenizers():

    breakpoint()
    assert sent_tokenize(MY_PREAMBLE) == ['Friends, Romans, countrymen, lend me your ears; 911']
    assert word_tokenize(MY_PREAMBLE) == [
        'Friends', ',', 'Romans', ',', 'countrymen', ',',
        'lend', 'me', 'your', 'ears', ';', '911'
    ]

    assert sent_tokenize(MY_MESSAGE) == [
        " Oh HeY there - so whatr'u  up to later????",
        'Statue of Liberty trip later.',
        'Text me (123) 456-7890. k cool!'
    ]
    assert word_tokenize(MY_MESSAGE) == [
        'Oh', 'HeY', 'there', '-', 'so', 'whatr', "'", 'u', 'up', 'to', 'later', '?', '?', '?', '?',
        'Statue', 'of', 'Liberty', 'trip', 'later', '.',
        'Text', 'me', '(', '123', ')', '456-7890.', 'k', 'cool', '!'
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
