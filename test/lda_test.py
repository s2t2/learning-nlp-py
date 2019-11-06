
import numpy as np

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from app.novels import token_stream
from conftest import TOKEN_SETS, NOVELS_DIRPATH

def test_lda_model():
    #generator = token_stream(NOVELS_DIRPATH)
    #dictionary = Dictionary(generator)

    dictionary = Dictionary(TOKEN_SETS)
    bags_of_words = [dictionary.doc2bow(tokens) for tokens in TOKEN_SETS]
    lda = LdaMulticore(corpus=bags_of_words, id2word=dictionary)

    response = lda.print_topics()
    assert isinstance(response, list)
    assert isinstance(response[0], tuple)
    assert isinstance(response[0][0], np.int64)
    assert isinstance(response[0][1], str)

    topic_strings = [topic_str for topic_str in response[0][1].split(" + ")]
    assert topic_strings[0] == '0.067*"sleep"'
    assert topic_strings[0].replace('"',"").split("*") == ['0.067', 'sleep']
