import os
#from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEWS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "amazon_reviews.csv")

if __name__ == "__main__":

    #
    # LOADING
    #

    df = pd.read_csv(REVIEWS_CSV_FILEPATH)

    #
    # SUMMARIZING
    #

    print(df[["reviews.rating", "reviews.text"]])

    print("REVIEW RATINGS...")
    print(df["reviews.rating"].value_counts())
    df["reviews.rating"].value_counts().sort_index().plot.bar()
    plt.show()

    print("REVIEW TEXT...")
    print(df["reviews.text"].str.len().value_counts())
    df["reviews.text"].str.len().plot.hist()
    plt.show()

    # PROCESSING

    breakpoint()

    # TODO: add column called "reviews.text_tokens" and add tokenized text there

    # TODO: add column called "reviews.text_vectors" and add vectorized tokens there

    # TODO: add column called "training.set" and split into subsets "test" and "train"
