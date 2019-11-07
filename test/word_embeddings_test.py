import os
#import itertools
from operator import itemgetter

from conftest import MY_PREAMBLE, MY_MESSAGE, DOCUMENTS, NOVELS_DIRPATH

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

def test_spacy_entities(nlp):
    doc = nlp(MY_PREAMBLE) #> <class 'spacy.tokens.doc.Doc'>
    entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]
    assert entities == [
        {'entity': 'Romans', 'label': 'NORP'},
        {'entity': '911', 'label': 'CARDINAL'}
    ]

    #CORPUS = " ".join(DOCUMENTS)
    #doc = nlp(CORPUS) #> <class 'spacy.tokens.doc.Doc'>
    #entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]
    #assert entities == []

    TXT_FILEPATH = os.path.join(NOVELS_DIRPATH, "Austen_Emma0000.txt")
    file_contents = open(TXT_FILEPATH).read().replace("\n", " ")
    doc = nlp(file_contents) #> <class 'spacy.tokens.doc.Doc'>
    entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]
    assert len(entities) == 56 # some are duplicates
    #assert entities[0] ==

    # aggregate and count number of appearances for each entity
    # h/t: https://github.com/prof-rossetti/nyu-info-2335-201905/blob/master/notes/python/modules/itertools.md
    #sorted_entities = sorted(entities, key=itemgetter("entity"))
    #grouped_entities = itertools.groupby(sorted_entities, key=itemgetter("entity"))
#
    #for entity, group in grouped_entities:
    #    print("----")
    #    print(entity)
    #    print(group)

    entity_names = [ent.text for ent in doc.ents]
    entity_counts = [{"entity": ent.text, "label": ent.label_, "count": entity_names.count(ent.text)} for ent in doc.ents]
    unique_entities = [dict(t) for t in {tuple(d.items()) for d in entity_counts}]
    sorted_entities = sorted(unique_entities, key=itemgetter("entity"), reverse=True) # prelim sort to ensure order is same as below assertion
    top_entities = sorted(sorted_entities, key=itemgetter("count"), reverse=True)[0:10]
    assert top_entities == [
        {'entity': 'Taylor', 'label': 'PERSON', 'count': 10},
        {'entity': 'Emma', 'label': 'PERSON', 'count': 7},
        {'entity': 'first', 'label': 'ORDINAL', 'count': 3},
        {'entity': 'Hartfield', 'label': 'PERSON', 'count': 3},
        {'entity': 'Woodhouse', 'label': 'PERSON', 'count': 2},
        {'entity': 'Weston', 'label': 'PERSON', 'count': 2},
        {'entity': 'Isabella', 'label': 'PERSON', 'count': 2},
        {'entity': 'years', 'label': 'DATE', 'count': 1},
        {'entity': 'two', 'label': 'CARDINAL', 'count': 1},
        {'entity': 'third', 'label': 'ORDINAL', 'count': 1}
    ]
