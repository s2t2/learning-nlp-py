
from conftest import MY_PREAMBLE, MY_MESSAGE

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

def test_spacy_noun_chunks(nlp):
    doc = nlp(MY_PREAMBLE) #> <class 'spacy.tokens.doc.Doc'>
    assert [span.text for span in doc.noun_chunks] == ['Friends', 'Romans', 'countrymen', 'me', 'your ears']

    doc = nlp(MY_MESSAGE) #> <class 'spacy.tokens.doc.Doc'>
    assert [span.text for span in doc.noun_chunks] == ['Statue', 'Liberty', 'Text', 'me', 'k']
