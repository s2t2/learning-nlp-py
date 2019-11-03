import pandas as pd
from sklearn.model_selection import train_test_split

from app.tokenizer import REVIEWS_CSV_FILEPATH
#from app.vectorizer import text_files_dataframe

#def balanced_reviews():


if __name__ == "__main__":

    df = pd.read_csv(REVIEWS_CSV_FILEPATH)
    x = df["reviews.text"] # inputs
    y = df["reviews.rating"] # outputs

    print("DOC LENGTHS:")
    print(x.str.len().value_counts())

    print("STAR-RATINGS:")
    print(y.value_counts())
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

    xtrain, xtest, ytrain, ytest = train_test_split(x.values, y.values, test_size=0.2, random_state=812)
    print(len(xtrain), len(xtest), len(ytrain), len(ytest)) #> 22665 5667 22665 5667
