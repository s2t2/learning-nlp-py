import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from app.tokenizer import REVIEWS_CSV_FILEPATH

if __name__ == "__main__":

    print("--------------------------")
    print("READING CSV OF AMAZON REVIEWS...")
    df = pd.read_csv(REVIEWS_CSV_FILEPATH)
    x = df["reviews.text"] # inputs
    y = df["reviews.rating"] # outputs
    print("DOC LENGTHS:")
    print(x.str.len().value_counts())
    print("STAR-RATINGS:")
    #print(y.value_counts())
    #> 5    19897
    #> 4     5648
    #> 3     1206
    #> 1      965
    #> 2      616
    print(y.value_counts(normalize=True).sort_index())
    #> 1    0.034060
    #> 2    0.021742
    #> 3    0.042567
    #> 4    0.199351
    #> 5    0.702280

    print("--------------------------")
    print("SPLITTING THE DATASET...")
    xtrain, xtest, ytrain, ytest = train_test_split(x.values, y.values, test_size=0.2, random_state=812)
    print(len(xtrain), len(xtest), len(ytrain), len(ytest)) #> 22665 5667 22665 5667

    print("--------------------------")
    print("VECTORIZING TRAINING DATA...")
    tv = TfidfVectorizer()
    tv.fit(xtrain)
    print("FEATURES", len(tv.get_feature_names())) #> 9621
    mtrain = tv.transform(xtrain)
    mtest = tv.transform(xtest)
    print("FEATURE MATRIX (TRAINING)", mtrain.shape) #> (22665, 9621)
    print("FEATURE MATRIX (TESTING)", mtrain.shape) #> (22665, 9621)

    print("--------------------------")
    print("LOGISTIC REGRESSION...")
    model = LogisticRegression(random_state=42)
    model.fit(mtrain, ytrain)
    ptrain = model.predict(mtrain)
    ptest = model.predict(mtest)
    print("PREDICTIONS (TRAINING)", ptrain.shape) #> (22665,)
    print("PREDICTIONS (TESTING)", ptest.shape) #> (5667,)
    atrain = accuracy_score(ytrain, ptrain)
    atest = accuracy_score(ytest, ptest)
    print("ACCY (TRAIN):", atrain) #> 0.7755129053606883
    print("ACCY (TEST):", atest) #> 0.7561319922357508
