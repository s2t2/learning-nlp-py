
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

DOCS1 = [
    "did not like that movie", "not a good movie", "popcorn smells good",
    "i like it", "lots of action", "very funny"
]

DOCS2 = ["good movie", "not a good movie", "did not like", "i like it", "good one"]

def test_tfidf():
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

    features2 = tfidf.fit_transform(DOCS2)
    assert tfidf.get_feature_names() == ['good movie', 'like', 'movie', 'not']
    assert features2.todense().shape == (5,4)
    #>matrix(
    #>    [[0.70710678, 0.        , 0.70710678, 0.        ],
    #>    [0.57735027, 0.        , 0.57735027, 0.57735027],
    #>    [0.        , 0.70710678, 0.        , 0.70710678],
    #>    [0.        , 1.        , 0.        , 0.        ],
    #>    [0.        , 0.        , 0.        , 0.        ]])
