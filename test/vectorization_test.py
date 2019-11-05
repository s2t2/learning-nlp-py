

from collections import Counter
from nltk.probability import FreqDist
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

def test_nltk_frequency_dist():
    TOKENS = np.concatenate(TOKEN_SETS)
    dist = FreqDist(TOKENS)
    assert dist.most_common(3) == [('all', 3), ('the', 2), ('kings', 2)]

def test_count_vectorizer():
    expected_features = ['all', 'and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz']
    expected_vals = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # "all the kings men"
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # "until they all got tired and went to sleep zzz"
    ])

    # fit then transform:
    cv = CountVectorizer()
    cv.fit(DOCUMENTS)
    assert cv.get_feature_names() == expected_features
    matrix = cv.transform(DOCUMENTS) #> <class 'scipy.sparse.csr.csr_matrix'>
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

    # fit and transform in the same step:
    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCUMENTS) #> <class 'scipy.sparse.csr.csr_matrix'>
    assert cv.get_feature_names() == expected_features
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_count_vectorizer_more():
    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCS1) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_features = ['action', 'did', 'funny', 'good', 'it', 'like', 'lots', 'movie', 'not', 'of', 'popcorn', 'smells', 'that', 'very']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], # "did not like that movie"
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # "not a good movie"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # "popcorn smells good"
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # "i like it"
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # "lots of action"
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # "very funny"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

    cv = CountVectorizer()
    matrix = cv.fit_transform(DOCS2) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_features = ['did', 'good', 'it', 'like', 'movie', 'not', 'one']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [0, 1, 0, 0, 1, 0, 0], # "good movie"
        [0, 1, 0, 0, 1, 1, 0], # "not a good movie"
        [1, 0, 0, 1, 0, 1, 0], # "did not like"
        [0, 0, 1, 1, 0, 0, 0], # "i like it"
        [0, 1, 0, 0, 0, 0, 1]  # "good one"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_count_vectorizer_vocab():
    vocab = ['hens', 'kings', 'men', 'sleep']
    cv = CountVectorizer(vocabulary=vocab) # pass vocab to specify desired features
    assert cv.get_feature_names() == vocab
    matrix = cv.fit_transform(DOCUMENTS) #> <class 'scipy.sparse.csr.csr_matrix'>
    expected_vals = np.array([
        [0, 1, 1, 0], # "all the kings men"
        [1, 1, 0, 0], # "ate all the kings hens"
        [0, 0, 0, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_count_vectorizer_stopwords():
    cv = CountVectorizer(stop_words="english")
    assert "all" in cv.get_stop_words()
    matrix = cv.fit_transform(DOCUMENTS)
    expected_features = ['ate', 'got', 'hens', 'kings', 'men', 'sleep', 'tired', 'went', 'zzz']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0, 0], # "all the kings men"
        [1, 0, 1, 1, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [0, 1, 0, 0, 0, 1, 1, 1, 1] # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)

def test_count_vectorizer_stattrim():
    cv = CountVectorizer(min_df=0.05, max_df=0.99) # only include features which occur in at least x% of the documents and at most y% of the documents. in this case, filters out "all"
    matrix = cv.fit_transform(DOCUMENTS)
    expected_features = ['and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # "all the kings men"
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_count_vectorizer_ngrams():
    cv = CountVectorizer(ngram_range=(1,2))
    matrix = cv.fit_transform(DOCUMENTS)
    expected_features = ['all', 'all got', 'all the', 'and', 'and went', 'ate', 'ate all', 'got', 'got tired', 'hens', 'kings', 'kings hens', 'kings men', 'men', 'sleep', 'sleep zzz', 'the', 'the kings', 'they', 'they all', 'tired', 'tired and', 'to', 'to sleep', 'until', 'until they', 'went', 'went to', 'zzz']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # "all the kings men"
        [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_count_vectorizer_custom(nlp):
    def my_tokenizer(text):
        doc = nlp(text) #> <class 'spacy.tokens.doc.Doc'>
        tokens = [token.lemma_.lower() for token in doc if token.is_stop == False and token.is_punct == False and token.is_space == False]
        return tokens

    #cv = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer, ngram_range=(1,2), stop_words="english")
    cv = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(1,2), stop_words="english")
    matrix = cv.fit_transform(DOCUMENTS)
    expected_features = ['eat', 'eat king', 'hen', 'king', 'king hen', 'king man', 'man', 'sleep', 'sleep zzz', 'tired', 'tired sleep', 'zzz']
    assert cv.get_feature_names() == expected_features
    expected_vals = np.array([
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], # "all the kings men"
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # "ate all the kings hens"
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # "until they all got tired and went to sleep zzz"
    ])
    assert np.array_equal(matrix.toarray(), expected_vals)
    assert np.array_equal(matrix.todense(), expected_vals)

def test_tfidf_vectorizer():
    tv = TfidfVectorizer()
    matrix = tv.fit_transform(DOCUMENTS)
    expected_features = ['all', 'and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz']
    assert tv.get_feature_names() == expected_features
    assert matrix.toarray().shape == (3, 15)
    assert matrix.todense().shape == (3, 15)
    expected_sparse_vals = [
        # "all the kings men"
        [0.3731, 0.0,   0.0,    0.0,    0.0,    0.4805, 0.6317, 0.0,    0.4805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "ate all the kings hens"
        [0.3154, 0.0,   0.5341, 0.0,    0.5341, 0.4062, 0.0,    0.0,    0.4062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "until they all got tired and went to sleep zzz"
        [0.1932, 0.3271, 0.0,   0.3271, 0.0,    0.0,    0.0,    0.3271, 0.0,    0.3271, 0.3271, 0.3271, 0.3271, 0.3271, 0.3271]
    ]
    sparse_vals = [
        [round(float(val), 4) for val in matrix.toarray()[0]],
        [round(float(val), 4) for val in matrix.toarray()[1]],
        [round(float(val), 4) for val in matrix.toarray()[2]],
    ] # convert floats for comparison purposes. can probably alternatively map the matrix
    assert np.array_equal(sparse_vals, expected_sparse_vals)
    # in this case, dense and sparse are not the same, as each dense row has the sparse row wrapped in a []
    assert matrix.toarray()[0].shape == (15,)
    assert matrix.todense()[0].shape == (1, 15)
    assert np.array_equal([matrix.toarray()[0]], matrix.todense()[0])

def test_tfidf_vectorizer_more():
    tv = TfidfVectorizer()
    matrix = tv.fit_transform(DOCS1) #> <6x4 sparse matrix of type '<class 'numpy.float64'>'
    expected_features = ['action', 'did', 'funny', 'good', 'it', 'like', 'lots', 'movie', 'not', 'of', 'popcorn', 'smells', 'that', 'very']
    assert tv.get_feature_names() == expected_features
    assert matrix.toarray().shape == (6,14)
    assert matrix.todense().shape == (6,14)
    #expected_sparse_vals = np.array([
    #    [0.        , 0.49892408, 0.        , 0.        , 0.        , 0.40912489, 0.        , 0.40912489, 0.40912489, 0.        , 0.        , 0.        , 0.49892408, 0.        ],
    #    [0.        , 0.        , 0.        , 0.57735027, 0.        , 0.        , 0.        , 0.57735027, 0.57735027, 0.        , 0.        , 0.        , 0.        , 0.        ],
    #    [0.        , 0.        , 0.        , 0.50161301, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.61171251, 0.61171251, 0.        , 0.        ],
    #    [0.        , 0.        , 0.        , 0.        , 0.77326237, 0.6340862 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
    #    [0.57735027, 0.        , 0.        , 0.        , 0.        , 0.        , 0.57735027, 0.        , 0.        , 0.57735027, 0.        , 0.        , 0.        , 0.        ],
    #    [0.        , 0.        , 0.70710678, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.70710678]
    #])
    #assert np.array_equal(matrix.toarray(), expected_sparse_vals)

    tv = TfidfVectorizer()
    matrix = tv.fit_transform(DOCS2)
    expected_features = ['did', 'good', 'it', 'like', 'movie', 'not', 'one']
    assert tv.get_feature_names() == expected_features
    assert matrix.toarray().shape == (5,7)
    assert matrix.todense().shape == (5,7)
    #expected_sparse_vals = np.array([
    #    [0.        , 0.63871058, 0.        , 0.        , 0.76944707, 0.        , 0.        ],
    #    [0.        , 0.50620441, 0.        , 0.        , 0.60981846, 0.60981846, 0.        ],
    #    [0.659118  , 0.        , 0.        , 0.53177225, 0.        , 0.53177225, 0.        ],
    #    [0.        , 0.        , 0.77828292, 0.62791376, 0.        , 0.        , 0.        ],
    #    [0.        , 0.55645052, 0.        , 0.        , 0.        , 0.        , 0.83088075]
    #])
    #assert np.array_equal(matrix.toarray(), expected_sparse_vals)

def test_tfidf_vectorizer_custom(nlp):
    def my_tokenizer(text):
        doc = nlp(text) #> <class 'spacy.tokens.doc.Doc'>
        tokens = [token.lemma_.lower() for token in doc if token.is_stop == False and token.is_punct == False and token.is_space == False]
        return tokens

    tv = TfidfVectorizer(tokenizer=my_tokenizer, min_df=0.25, max_df=0.95, ngram_range=(1,2))
    matrix = tv.fit_transform(DOCUMENTS)
    expected_features = ['eat', 'eat king', 'get', 'get tired', 'go', 'go sleep', 'hen', 'king', 'king hen', 'king man', 'man', 'sleep', 'sleep zzz', 'tired', 'tired go', 'zzz']
    assert tv.get_feature_names() == expected_features
    #expected_sparse_vals = np.array([
    #    # "all the kings men"
    #    [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.4736296 , 0.        , 0.62276601, 0.62276601, 0.        , 0.        , 0.        , 0.        , 0.        ],
    #    # "ate all the kings hens"
    #    [0.46735098, 0.46735098, 0.        , 0.        , 0.        , 0.        , 0.46735098, 0.35543247, 0.46735098, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
    #    # "until they all got tired and went to sleep zzz"
    #    [0.        , 0.        , 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333]
    #])
    #sparse_vals = [
    #    [round(float(val), 4) for val in matrix.toarray()[0]],
    #    [round(float(val), 4) for val in matrix.toarray()[1]],
    #    [round(float(val), 4) for val in matrix.toarray()[2]],
    #]
    #assert np.array_equal(sparse_vals, expected_sparse_vals)
    assert matrix.toarray().shape == (3, 16)
    assert matrix.todense().shape == (3, 16)
    assert matrix.toarray()[0].shape == (16,)
    assert matrix.todense()[0].shape == (1, 16)

def test_spacy_vectors(nlp):
    doc = nlp("Two bananas in pyjamas")
    matrix = doc.vector
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (300,)
