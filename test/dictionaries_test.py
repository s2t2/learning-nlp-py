#
# gensim.corpora.Dictionary
#   + encapsulates the mapping between normalized words and their integer ids
#   + defines the vocabulary of all words that our processing knows about
#
from gensim.corpora import Dictionary

from conftest import TOKEN_SETS

def test_gensim_dictionaries():

    dictionary = Dictionary(TOKEN_SETS)

    token_items = list(dictionary.items())
    assert token_items == [(0, 'all'), (1, 'kings'), (2, 'men'), (3, 'the'), (4, 'ate'), (5, 'hens'), (6, 'and'), (7, 'got'), (8, 'sleep'), (9, 'they'), (10, 'tired'), (11, 'to'), (12, 'until'), (13, 'went'), (14, 'zzz')]

    tokens_map = dictionary.token2id
    assert tokens_map == {'all': 0, 'kings': 1, 'men': 2, 'the': 3, 'ate': 4, 'hens': 5, 'and': 6, 'got': 7, 'sleep': 8, 'they': 9, 'tired': 10, 'to': 11, 'until': 12, 'went': 13, 'zzz': 14}

    token_counts = [dictionary.doc2bow(token_set) for token_set in TOKEN_SETS]
    assert token_counts == [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 1), (1, 1), (3, 1), (4, 1), (5, 1)],
        [(0, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1)]
    ]
