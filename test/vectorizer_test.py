import os
import pandas as pd

from app.vectorizer import count_vectorized_dataframe, tfidf_vectorized_dataframe, cosine_similarity_dataframe
from conftest import DOCUMENTS

MOCK_EXPORTS_DIRPATH = os.path.join(os.path.dirname(__file__), "data", "exports")

texts_df = pd.DataFrame({"txt.filename": ["Doc 0", "Doc 1", "Doc 2"], "txt.contents": DOCUMENTS})

def test_count_vectorized_dataframe():
    df = count_vectorized_dataframe(texts_df)
    df.to_csv(os.path.join(MOCK_EXPORTS_DIRPATH, "counts_matrix.csv"))
    assert df.shape == (3, 17)
    assert df.columns.tolist() == [
        'txt.filename', 'txt.contents', 'all', 'and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz'
    ]
    assert df.iloc[0].values.tolist() == [
        'Doc 0', 'all the kings men', 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0
    ]

def test_tfidf_vectorized_dataframe():
    df = tfidf_vectorized_dataframe(texts_df)
    df.to_csv(os.path.join(MOCK_EXPORTS_DIRPATH, "tfidf_matrix.csv"))
    assert df.shape == (3, 17)
    assert df.columns.tolist() == [
        'txt.filename', 'txt.contents', 'all', 'and', 'ate', 'got', 'hens', 'kings', 'men', 'sleep', 'the', 'they', 'tired', 'to', 'until', 'went', 'zzz'
    ]
    assert df.iloc[0].values.tolist() == [
        'Doc 0', 'all the kings men', 0.3731188059313277, 0.0, 0.0, 0.0, 0.0, 0.4804583972923858, 0.6317450542765208, 0.0, 0.4804583972923858, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

def test_cosine_similarity_dataframe():
    df = tfidf_vectorized_dataframe(texts_df)
    similarities_df = cosine_similarity_dataframe(df)
    similarities_df.to_csv(os.path.join(MOCK_EXPORTS_DIRPATH, "tfidf_cosine_similarities.csv"))
    assert similarities_df.shape == (3, 5)
    assert similarities_df.columns.tolist() == ['txt.filename', 'txt.contents', 0, 1, 2]
    assert similarities_df.to_dict("records") == [
        {
            'txt.filename': 'Doc 0',
            'txt.contents': 'all the kings men',
            0: 1.0000000000000002,
            1: 0.5080146464112386,
            2: 0.0720732085701743
        }, {
            'txt.filename': 'Doc 1',
            'txt.contents': 'ate all the kings hens',
            0: 0.5080146464112386,
            1: 1.0000000000000002,
            2: 0.06093252799951177
        }, {
            'txt.filename':'Doc 2',
            'txt.contents': 'until they all got tired and went to sleep zzz',
            0: 0.0720732085701743,
            1: 0.06093252799951177,
            2: 1.0000000000000002
        }]


def test_spacy_similarity(nlp):
    doc1 = nlp(DOCUMENTS[0])
    doc2 = nlp(DOCUMENTS[1])
    doc3 = nlp(DOCUMENTS[2])

    assert doc1.similarity(doc2) == 0.7986792558854244
    assert doc1.similarity(doc3) == 0.6597250374980631
    assert doc2.similarity(doc3) == 0.712800124804148
