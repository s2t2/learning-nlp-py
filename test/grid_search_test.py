import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

# todo: consider moving into a fixture or something
from app.whiskey_reviews_classifier import WHISKEY_DATA_DIRPATH
training_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "train.csv"))
xtrain = training_df["description"]
ytrain = training_df["category"]

def test_pipeline():
    tv = TfidfVectorizer()
    rf = RandomForestClassifier()
    pipeline = Pipeline([("vect", tv), ("clf", rf)])

    param_grid = {
        "vect__stop_words": [None, "english"]
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

    grid_search.fit(xtrain, ytrain)
    print("PIPELINE SCORE", grid_search.best_score_)
    assert grid_search.best_score_ > 0.8
    assert grid_search.best_params_ == {'vect__stop_words': 'english'}

def test_nested_pipeline():
    tv = TfidfVectorizer()
    svd = TruncatedSVD(n_components=100, algorithm="randomized", n_iter=10)
    lsi = Pipeline([("vect", tv), ("svd", svd)])
    rf = RandomForestClassifier()
    pipeline = Pipeline([("lsi", lsi), ("clf", rf)])

    param_grid = {
        "lsi__svd__n_components": [10, 100, 250],
        "lsi__vect__max_df": [0.9, 0.95, 1.0],
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=4, verbose=0)

    grid_search.fit(xtrain, ytrain)
    print("NESTED PIPELINE SCORE", grid_search.best_score_)
    assert grid_search.best_score_ > 0.85
    #assert grid_search.best_params_ == {'lsi__svd__n_components': 10, 'lsi__vect__max_df': 0.95}
