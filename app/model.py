

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups


if __name__ == "__main__":
    print("MY MODEL(S)...")

    # Tune a vectorizer within a document classification pipeline
    # Apply a vectorization method to a document classification problem
    # Benchmark and compare various vectorization methods in document classification tasks

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
    print("ALL CATEGORIES...", ALL_CATEGORIES)

    categories = ["sci.space", "sci.med", "sci.electronics"]
    data = fetch_20newsgroups(subset="train", categories=categories)
    #> Downloading 20news dataset. This may take a few minutes.
    #> Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)

    #breakpoint()

    #print(type(data)) #> <class 'sklearn.utils.Bunch'>
    print(dir(data)) #> ['DESCR', 'data', 'filenames', 'target', 'target_names']
    print(data.data[100]) #> looks like an email hmm...
