import os
#from pprint import pprint

import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import squarify

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEWS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "amazon_reviews.csv")

ALPHANUMERIC_PATTERN = r'[^a-zA-Z ^0-9]' # same as "[^a-zA-Z ^0-9]"

def tokenize(doc):
    """
    Params: doc (str)
    """
    doc = doc.lower() # normalize case
    doc = re.sub(ALPHANUMERIC_PATTERN, "", doc) # keep only alphanumeric characters
    tokens = doc.split()
    # todo: consider removing stopwords!
    # todo: consider stemming / lemmatizing!
    return tokens

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

    #
    # PROCESSING > TOKENIZING
    #

    df["nlp.tokens"] = df["reviews.text"].apply(lambda txt: tokenize(txt))

    print("TOKENS...")
    print(df["nlp.tokens"].head())

    tokens = np.hstack(df["nlp.tokens"].values) # h/t: https://stackoverflow.com/a/11367444/670433

    #sns.countplot(tokens)
    #plt.show()
    #> hmmm this is hanging pretty bad

    #sns.countplot(["all", "the", "kings", "men", "ate", "all", "the", "kings", "hens"])
    #plt.show()



    #summary_df = pd.DataFrame()
    #most_frequent_tokens = summary_df[summary_df["rank"] <= 20]
#
    #squarify.plot(sizes=summary_df["pct_total"], label=summary_df["word"], alpha=0.8 )
    #plt.axis("off")
    #plt.show()









    #
    # PROCESSING > VECTORIZING
    #

    # TODO: add column called "nlp.vectors" and add vectorized tokens there

    #
    # PROCESSING > SUBSETS
    #

    # TODO: add column called "nlp.subset" and split into subsets "test" and "train"
