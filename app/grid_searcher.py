

# Tune a vectorizer within a document classification pipeline
# Apply a vectorization method to a document classification problem
# Benchmark and compare various vectorization methods in document classification tasks

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups

def download_data(categories=None):
    ALL_CATEGORIES = [
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "misc.forsale",
        "talk.politics.misc",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.religion.misc",
        "alt.atheism",
        "soc.religion.christian",
    ] # h/t: http://qwone.com/~jason/20Newsgroups/
    #print("ALL CATEGORIES...", ALL_CATEGORIES)

    if not categories:
        #categories = ["alt.atheism", "talk.religion.misc"]
        categories = ["sci.space", "sci.med", "sci.electronics"]

    data = fetch_20newsgroups(subset="train", categories=categories)
    #> Downloading 20news dataset. This may take a few minutes.
    #> Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)
    #breakpoint()
    #print(type(data)) #> <class 'sklearn.utils.Bunch'>
    #print(dir(data)) #> ['DESCR', 'data', 'filenames', 'target', 'target_names']
    #print(data.data[100]) #> looks like an email hmm...
    return data

def my_grid_search():
    # Estimator
    # This is assumed to implement the scikit-learn estimator interface.
    # Either estimator needs to provide a score function, or scoring must be passed.
    tv = TfidfVectorizer(stop_words="english")
    model = RandomForestClassifier()
    pipe = Pipeline([('vect', tv), ('clf', model)])

    # Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or
    # a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
    # This enables searching over any sequence of parameter settings.
    params_grid = {
        "vect__min_df": (0.02, 0.05),
        "vect__max_df": (0.75, 1.0),
        "vect__max_features": (500, 1000),
        "clf__n_estimators": (5, 10),
        "clf__max_depth": (15, 20)
    }

    # Cross-validation generator
    # Determines the cross-validation splitting strategy.
    # Possible inputs for cv are:
    #   + None, to use the default 3-fold cross validation,
    #   + integer, to specify the number of folds in a (Stratified)KFold,
    #   + CV splitter,
    #   + An iterable yielding (train, test) splits as arrays of indices.
    cv = 5

    # Number of jobs to run in parallel.
    #   + None means 1 unless in a joblib.parallel_backend context.
    #   + -1 means using all processors. See Glossary for more details.
    n_jobs = -1

    # GridSearchCV exhaustively generates candidates from a grid of parameter values
    grid_search = GridSearchCV(estimator=pipe, param_grid=params_grid, cv=cv, n_jobs=n_jobs, verbose=1)

    return grid_search

if __name__ == "__main__":

    # initialization
    grid_search = my_grid_search()

    # training / tuning
    training_data = download_data()
    grid_search.fit(training_data.data, training_data.target)
    print(grid_search.best_score_) #> 0.8689538807649044

    # prediction / classification
    xtest = [
        "Astronauts pass the moons of Jupiter in their space shuttle",
        "Hey I'm talking about basketball here"
    ]
    results = grid_search.predict(xtest)
    print(results) #> [2, 0]
