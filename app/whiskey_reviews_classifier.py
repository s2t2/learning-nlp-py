
# FYI CATEGORIES = {1: "Scotch", 2: "Bourbon/Tennessee", 3: "Craft Whiskey", 4: "Canadian"}

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

WHISKEY_DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "whiskey")
SUBMISSION_CSV_FILEPATH = os.path.join(WHISKEY_DATA_DIRPATH, "my_submission.csv")
GS_SUBMISSION_CSV_FILEPATH = os.path.join(WHISKEY_DATA_DIRPATH, "my_gs_submission.csv")

def generate_model_submissions(vectorizer, model, submission_csv_filepath=SUBMISSION_CSV_FILEPATH):
    """
    Uses a given pre-trained model to classify values in the testing dataset,
    and writes the results to CSV in the following format:
        id,category
        955,1
        3532,3
        1390,2
        1024,4
    """
    # use the model to predict categories for all texts in the testing set
    testing_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "test.csv"))
    xtest = testing_df["description"]
    matrix = vectorizer.transform(xtest)
    predictions = model.predict(matrix.todense())

    # then write those to SUBMISSION_CSV_FILEPATH, and return the dataframe for good measure
    submission_df = pd.DataFrame({"id": testing_df["id"], "category": predictions})
    submission_df["category"] = submission_df["category"].astype("int64") # convert to ints
    submission_df.to_csv(submission_csv_filepath, index=False)
    return submission_df

def generate_grid_search_submissions(grid_search, submission_csv_filepath=GS_SUBMISSION_CSV_FILEPATH):
    """
    Uses a given pre-trained grid search to find the best params,
    then use the best version of the model to classify values in the testing dataset,
    and writes the results to CSV in the following format:
        id,category
        955,1
        3532,3
        1390,2
        1024,4
    """
    # use the model to predict categories for all texts in the testing set
    testing_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "test.csv"))
    xtest = testing_df["description"]
    predictions = grid_search.predict(xtest)

    # then write those to SUBMISSION_CSV_FILEPATH, and return the dataframe for good measure
    submission_df = pd.DataFrame({"id": testing_df["id"], "category": predictions})
    submission_df["category"] = submission_df["category"].astype("int64") # convert to ints
    submission_df.to_csv(submission_csv_filepath, index=False)
    return submission_df

if __name__ == "__main__":

    training_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "train.csv"))
    print(training_df.head())
    #print(training_df["category"].value_counts())
    #> 1    1637
    #> 2     449
    #> 3     300
    #> 4     200

    #for i, row in list(training_df.iterrows())[0:4]:
    #    print(row["id"], row["category"], row["description"][0:90])

    xtrain = training_df["description"]
    ytrain = training_df["category"]

    #
    # SINGLE MODEL APPROACH
    #

    tv = TfidfVectorizer()
    tv.fit(xtrain)
    matrix = tv.transform(xtrain)

    rf = RandomForestClassifier()
    rf.fit(matrix.todense(), ytrain)
    predictions = rf.predict(matrix)

    accy = accuracy_score(ytrain, predictions)
    print("TRAINING ACCY:", accy)

    submission_df = generate_model_submissions(tv, rf)
    print("SUBMISSION FILE...")
    print(submission_df.head())

    #
    # GRID SEARCH APPROACH
    #

    tv = TfidfVectorizer()
    rf = RandomForestClassifier()
    pipeline = Pipeline([("vect", tv), ("clf", rf)])
    param_grid = {
        "vect__stop_words": [None, "english"],
        #"vect__min_df": (0.02, 0.05),
        #"vect__max_df": (0.75, 1.0),
        #"vect__max_features": (500, 1000),
        #"clf__n_estimators": (5, 10),
        #"clf__max_depth": (15, 20)
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=10)

    grid_search.fit(xtrain, ytrain)
    print("BEST SCORE:", grid_search.best_score_) #> ____
    print("BEST PARAMS:", grid_search.best_params_) #> ______

    submission_df = generate_grid_search_submissions(grid_search)
    print("SUBMISSION FILE...")
    print(submission_df.head())
