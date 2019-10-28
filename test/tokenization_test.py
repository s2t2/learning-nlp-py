from spacy.tokenizer import Tokenizer
from nltk.tokenize import sent_tokenize # Sentence Tokenizer
from nltk.tokenize import word_tokenize # Word Tokenizer

from conftest import MY_PREAMBLE, MY_MESSAGE

def test_nltk_sentence_tokenizer():
    assert sent_tokenize(MY_PREAMBLE) == [
        'Friends, Romans, countrymen, lend me your ears; 911'
    ]

    assert sent_tokenize(MY_MESSAGE) == [
        " Oh HeY there - so whatr'u  up to later????",
        'Statue of Liberty trip later.',
        'Text me (123) 456-7890. k cool!'
    ]

def test_nltk_word_tokenizer():
    assert word_tokenize(MY_PREAMBLE) == [
        'Friends', ',', 'Romans', ',', 'countrymen', ',',
        'lend', 'me', 'your', 'ears', ';', '911'
    ]

    assert word_tokenize(MY_MESSAGE) == [
        'Oh', 'HeY', 'there', '-', 'so', 'whatr', "'", 'u', 'up', 'to', 'later', '?', '?', '?', '?',
        'Statue', 'of', 'Liberty', 'trip', 'later', '.',
        'Text', 'me', '(', '123', ')', '456-7890.', 'k', 'cool', '!'
    ]

def test_spacy_tokenizer(nlp):
    tokenizer = Tokenizer(nlp.vocab)

    tokens = [token.text for token in tokenizer(MY_PREAMBLE)]
    assert tokens == ['Friends,', 'Romans,', 'countrymen,', 'lend', 'me', 'your', 'ears;', '911']

    tokens = [token.text for token in tokenizer(MY_MESSAGE)]
    assert tokens == [
        ' ', 'Oh', 'HeY', 'there', '-', 'so', "whatr'u", ' ', 'up', 'to', 'later????', '\n   ',
        'Statue', 'of', 'Liberty', 'trip', 'later.', '\n ',
        'Text', 'me', '(123)', '456-7890.', 'k', 'cool!'
    ]
