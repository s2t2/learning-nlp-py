
import numpy as np

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import TfidfModel

from app.novels import token_stream, parse_topics
from conftest import TOKEN_SETS, NOVELS_DIRPATH

def test_lda_model():
    dictionary = Dictionary(TOKEN_SETS)
    bags_of_words = [dictionary.doc2bow(tokens) for tokens in TOKEN_SETS]
    lda = LdaMulticore(corpus=bags_of_words, id2word=dictionary, random_state=723812, passes=10, workers=4)
    response = lda.print_topics()
    assert isinstance(response, list)
    assert isinstance(response[0], tuple)
    assert isinstance(response[0][0], np.int64)
    assert isinstance(response[0][1], str)
    topic_strings = [topic_str for topic_str in response[0][1].split(" + ")]
    assert topic_strings[0] == '0.067*"sleep"'

    #top_topics = lda.top_topics(bags_of_words)
    #assert top_topics[0][0] == [
    #    (0.09950739, 'all'), (0.09950739, 'and'), (0.09950739, 'got'), (0.09950739, 'sleep'),
    #    (0.09950739, 'they'), (0.09950739, 'tired'), (0.09950739, 'to'), (0.09950739, 'until'),
    #    (0.09950739, 'went'), (0.09950739, 'zzz'), (0.0009852239, 'kings'), (0.0009852239, 'men'),
    #    (0.0009852239, 'the'), (0.0009852239, 'ate'), (0.0009852239, 'hens')
    #]




def test_lda_model_tfidf():
    dictionary = Dictionary(TOKEN_SETS)
    bags_of_words = [dictionary.doc2bow(tokens) for tokens in TOKEN_SETS]
    tfidf = TfidfModel(bags_of_words, normalize=True)
    tfidf_bags = tfidf[bags_of_words]
    lda = LdaMulticore(corpus=tfidf_bags, id2word=dictionary, random_state=723812, passes=10, workers=4)
    response = lda.print_topics()
    assert isinstance(response, list)
    assert isinstance(response[0], tuple)
    assert isinstance(response[0][0], np.int64)
    assert isinstance(response[0][1], str)
    topic_strings = [topic_str for topic_str in response[0][1].split(" + ")]
    assert topic_strings[0] == '0.067*"sleep"'

def test_parse_topics():
    dictionary = Dictionary(TOKEN_SETS)
    bags_of_words = [dictionary.doc2bow(token_set) for token_set in TOKEN_SETS]
    lda = LdaMulticore(corpus=bags_of_words, id2word=dictionary)
    parsed_topics = parse_topics(lda)
    assert  parsed_topics[0] == {'sleep': 0.067, 'got': 0.067, 'went': 0.067, 'until': 0.067, 'to': 0.067, 'tired': 0.067, 'they': 0.067, 'all': 0.067, 'ate': 0.067, 'the': 0.067}

def test_lda_streaming():
    generator = token_stream(NOVELS_DIRPATH)
    dictionary = Dictionary(generator)
    bags_of_words = [dictionary.doc2bow(tokens) for tokens in generator]
    lda = LdaMulticore(corpus=bags_of_words, id2word=dictionary, random_state=723812, passes=10, workers=4)
    parsed_topics = parse_topics(lda)
    assert len(parsed_topics) == 20
    #print(parsed_topics[0])
    #assert parsed_topics[0] == {'upshe': 0.001, 'jane': 0.001, 'think': 0.001, 'regular': 0.001, 'facile': 0.001, 'her': 0.001, 'power': 0.001, 'intimate': 0.001, 'saythat': 0.001, 'manyacquaintance': 0.001}
    #assert parsed_topics[0] == {'quite': 0.001, 'pomps': 0.001, 'inutility': 0.001, 'counts': 0.001, 'brought': 0.001, 'repent': 0.001, 'dayabout': 0.001, 'professor': 0.001, 'upward': 0.001, 'been': 0.001}
    # results not repeatable unless you set the random_state param!
    assert parsed_topics[0] == {'doctor': 0.001, 'companion': 0.001, 'lucky': 0.001, 'somewhat': 0.001, 'ofchildhood': 0.001, 'rub': 0.001, 'idea': 0.001, 'pleasure': 0.001, 'ofexistence': 0.001, 'disposition': 0.001}
