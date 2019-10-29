
from conftest import MY_PREAMBLE

def test_spacy_parts_of_speech(nlp):
    doc = nlp("Hello World!!!") #> <class 'spacy.tokens.doc.Doc'>
    assert [str(token) for token in doc] == ["Hello", "World", "!", "!", "!"]
    assert [token.pos_ for token in doc] == ['INTJ', 'PROPN', 'PUNCT', 'PUNCT', 'PUNCT']

    doc = nlp(MY_PREAMBLE) #> <class 'spacy.tokens.doc.Doc'>
    assert [str(token) for token in doc] == [
        'Friends', ',', 'Romans', ',', 'countrymen', ',',
        'lend', 'me', 'your', 'ears', ';', '911'
    ]
    assert [token.pos_ for token in doc] == [
        'NOUN', 'PUNCT', 'PROPN', 'PUNCT', 'NOUN', 'PUNCT',
        'VERB', 'PRON', 'PRON', 'NOUN', 'PUNCT', 'NUM'
    ]
