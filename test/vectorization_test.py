

from collections import Counter
#from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

from conftest import TOKEN_SETS, DOCUMENTS

DOCS1 = [
    "did not like that movie", "not a good movie", "popcorn smells good",
    "i like it", "lots of action", "very funny"
]

DOCS2 = ["good movie", "not a good movie", "did not like", "i like it", "good one"]

def test_counter():
    counter = Counter()
    for tokens in TOKEN_SETS:
        counter.update(tokens) # pass a list of words to group by word, pass a word to group by char
    assert counter.most_common(3) == [('all', 3), ('the', 2), ('kings', 2)]

#def test_nltk_frequency_dist():
#    my_dist = FreqDist({'the': 3, 'dog': 2, 'not': 1})
#    print(my_dist.most_common(2))

def test_count_vectorizer():
    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCUMENTS) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_vals = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # "all the kings men"
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    features = cv.get_feature_names()
    assert features == ['all', 'and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz']

    vocab = ['hens', 'kings', 'men', 'sleep']
    cv = CountVectorizer(vocabulary=vocab) # pass vocab to specify desired features
    matrix = cv.fit_transform(DOCUMENTS) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_vals = np.array([
        [0, 1, 1, 0], # "all the kings men"
        [1, 1, 0, 0], # "ate all the kings hens"
        [0, 0, 0, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert cv.get_feature_names() == vocab

    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCS1) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_vals = np.array([
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], # "did not like that movie"
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # "not a good movie"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # "popcorn smells good"
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # "i like it"
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # "lots of action"
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # "very funny"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert cv.get_feature_names() == ['action', 'did', 'funny', 'good', 'it', 'like', 'lots', 'movie', 'not', 'of', 'popcorn', 'smells', 'that', 'very']

    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCS2) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_vals = np.array([
        [0, 1, 0, 0, 1, 0, 0], # "good movie"
        [0, 1, 0, 0, 1, 1, 0], # "not a good movie"
        [1, 0, 0, 1, 0, 1, 0], # "did not like"
        [0, 0, 1, 1, 0, 0, 0], # "i like it"
        [0, 1, 0, 0, 0, 0, 1]  # "good one"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert cv.get_feature_names() == ['did', 'good', 'it', 'like', 'movie', 'not', 'one']

def test_tfidf_vectorizer():
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))

    features = tfidf.fit_transform(DOCS1) #> <6x4 sparse matrix of type '<class 'numpy.float64'>'
    assert tfidf.get_feature_names() == ['good', 'like', 'movie', 'not']
    assert features.todense().shape == (6,4) #> ( len(DOCUMENTS), len(feature_names)  )
    #>matrix(
    #>   [[0.        , 0.57735027, 0.57735027, 0.57735027],
    #>    [0.57735027, 0.        , 0.57735027, 0.57735027],
    #>    [1.        , 0.        , 0.        , 0.        ],
    #>    [0.        , 1.        , 0.        , 0.        ],
    #>    [0.        , 0.        , 0.        , 0.        ],
    #>    [0.        , 0.        , 0.        , 0.        ]])

    features = tfidf.fit_transform(DOCS2)
    assert tfidf.get_feature_names() == ['good movie', 'like', 'movie', 'not']
    assert features.todense().shape == (5,4)
    #>matrix(
    #>    [[0.70710678, 0.        , 0.70710678, 0.        ],
    #>    [0.57735027, 0.        , 0.57735027, 0.57735027],
    #>    [0.        , 0.70710678, 0.        , 0.70710678],
    #>    [0.        , 1.        , 0.        , 0.        ],
    #>    [0.        , 0.        , 0.        , 0.        ]])
