
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from app.novels import token_stream
from conftest import TOKEN_SETS, NOVELS_DIRPATH

#
# gensim.corpora.Dictionary
#   + encapsulates the mapping between normalized words and their integer ids
#   + defines the vocabulary of all words that our processing knows about
#

NEW_TOKENS = ["all", "kings", "queens", "jacks"]

def test_dictionaries():
    dictionary = Dictionary(TOKEN_SETS)
    # it maps tokens to numeric indices:
    assert list(dictionary.items()) == [(0, 'all'), (1, 'kings'), (2, 'men'), (3, 'the'), (4, 'ate'), (5, 'hens'), (6, 'and'), (7, 'got'), (8, 'sleep'), (9, 'they'), (10, 'tired'), (11, 'to'), (12, 'until'), (13, 'went'), (14, 'zzz')]
    assert dictionary.token2id == {'all': 0, 'kings': 1, 'men': 2, 'the': 3, 'ate': 4, 'hens': 5, 'and': 6, 'got': 7, 'sleep': 8, 'they': 9, 'tired': 10, 'to': 11, 'until': 12, 'went': 13, 'zzz': 14}

def test_lookups():
    dictionary = Dictionary(TOKEN_SETS)
    # it can perform token inclusion and position lookups:
    assert dictionary.doc2idx(TOKEN_SETS[0]) == [0, 3, 1, 2] # ["all", "the", "kings", "men"]
    assert dictionary.doc2idx(NEW_TOKENS) == [0, 1, -1, -1] # ["all", "kings", "queens", "jacks"]

def test_statistical_trimming():
    dictionary = Dictionary(TOKEN_SETS)
    # no_below and no_above like min_df and max_df, except...
    #   + no_below: absolute number of documents
    #   + no_above: percentage of documents
    dictionary.filter_extremes(no_below=2, no_above=0.99)
    # it excludes terms not meeting the filter conditions:
    assert list(dictionary.items()) == [(0, 'kings'), (1, 'the')]
    assert dictionary.token2id == {'kings': 0, 'the': 1}

def test_vectorization_counts():
    dictionary = Dictionary(TOKEN_SETS)
    # it generates a count-vectorized bag of words for a given token set:
    bags_of_words = [dictionary.doc2bow(token_set) for token_set in TOKEN_SETS]
    assert bags_of_words == [
        # ["all", "the", "kings", "men"]
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        # ["ate", "all", "the", "kings", "hens"]
        [(0, 1), (1, 1), (3, 1), (4, 1), (5, 1)],
        # ["until", "they", "all", "got", "tired", "and", "went", "to", "sleep", "zzz"]
        [(0, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1)]
    ]

def test_vectorization_tfidf():
    dictionary = Dictionary(TOKEN_SETS)
    bags_of_words = [dictionary.doc2bow(token_set) for token_set in TOKEN_SETS]

    # it generates a tfidf-vectorized bag of words for a given token set:
    tm = TfidfModel(bags_of_words, normalize=True)
    tc = tm[bags_of_words]
    #print(type(transformed_corpus)) #> gensim.interfaces.TransformedCorpus
    assert list(tc) == [
        # ["all", "the", "kings", "men"]
        [(1, 0.32718457421366), (2, 0.8865102981879298), (3, 0.32718457421366)],
        # ["ate", "all", "the", "kings", "hens"]
        [(1, 0.2448297500958463), (3, 0.2448297500958463), (4, 0.6633689723434505), (5, 0.6633689723434505)],
        # ["until", "they", "all", "got", "tired", "and", "went", "to", "sleep", "zzz"]
        [(6, 0.3333333333333333), (7, 0.3333333333333333), (8, 0.3333333333333333), (9, 0.3333333333333333), (10, 0.3333333333333333), (11, 0.3333333333333333), (12, 0.3333333333333333), (13, 0.3333333333333333), (14, 0.3333333333333333)]
    ]

    # alternate construction:
    tm2 = TfidfModel(dictionary=dictionary)
    tc2 = tm2[bags_of_words]
    assert list(tc2) == [
        # ["all", "the", "kings", "men"]
        [(1, 0.32718457421366), (2, 0.8865102981879298), (3, 0.32718457421366)],
        # ["ate", "all", "the", "kings", "hens"]
        [(1, 0.2448297500958463), (3, 0.2448297500958463), (4, 0.6633689723434505), (5, 0.6633689723434505)],
        # ["until", "they", "all", "got", "tired", "and", "went", "to", "sleep", "zzz"]
        [(6, 0.3333333333333333), (7, 0.3333333333333333), (8, 0.3333333333333333), (9, 0.3333333333333333), (10, 0.3333333333333333), (11, 0.3333333333333333), (12, 0.3333333333333333), (13, 0.3333333333333333), (14, 0.3333333333333333)]
    ]

def test_streaming():
    generator = token_stream(NOVELS_DIRPATH)
    # it can be constructed via a generator:
    dictionary = Dictionary(generator)
    token_items = list(dictionary.items())
    assert len(token_items) == 1969
    assert token_items[0:4] == [(0, 'a'), (1, 'about'), (2, 'accommodate'), (3, 'admire')]
