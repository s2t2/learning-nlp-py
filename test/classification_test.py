
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# todo: consider moving into a fixture or something
import os
import pandas as pd
from app.whiskey_reviews_classifier import WHISKEY_DATA_DIRPATH
# ...
training_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "train.csv"))
xtrain = training_df["description"]
ytrain = training_df["category"]
# ...
testing_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "test.csv"))
xtest = testing_df["description"]
# ...
tv = TfidfVectorizer()
tv.fit(xtrain)
training_matrix = tv.transform(xtrain)
testing_matrix = tv.transform(xtest)
# end fixtures

def test_lr():
    lr = LogisticRegression(random_state=42)
    lr.fit(training_matrix.todense(), ytrain)

    training_predictions = lr.predict(training_matrix)
    assert training_predictions.shape == (2586,)
    training_accy = accuracy_score(ytrain, training_predictions)
    assert training_accy > 0.9

    testing_predictions = lr.predict(testing_matrix)
    assert testing_predictions.shape == (288,)

def test_nb():
    nb = MultinomialNB()
    nb.fit(training_matrix.todense(), ytrain)

    training_predictions = nb.predict(training_matrix)
    assert training_predictions.shape == (2586,)
    training_accy = accuracy_score(ytrain, training_predictions)
    assert training_accy > 0.65

    testing_predictions = nb.predict(testing_matrix)
    assert testing_predictions.shape == (288,)

def test_rf():
    rf = RandomForestClassifier()
    rf.fit(training_matrix.todense(), ytrain)

    training_predictions = rf.predict(training_matrix)
    assert training_predictions.shape == (2586,)
    training_accy = accuracy_score(ytrain, training_predictions)
    assert training_accy > 0.98

    testing_predictions = rf.predict(testing_matrix)
    assert testing_predictions.shape == (288,)
