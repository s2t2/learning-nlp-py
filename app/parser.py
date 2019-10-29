import os
#from pprint import pprint
from collections import Counter

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

def summarize(token_sets):
    """
    Param: token_sets a list of token lists
    """

    token_counter = Counter()
    doc_counter = Counter()

    for tokens in token_sets:
        token_counter.update(tokens)
        doc_counter.update(set(tokens)) # removes duplicate tokens so they only get counted once per doc!

    token_counts = zip(token_counter.keys(), token_counter.values())
    doc_counts = zip(doc_counter.keys(), doc_counter.values())

    # assemble dataframe of token stats

    df = pd.DataFrame(token_counts, columns=["token", "token_count"])

    df["token_rank"] = df["token_count"].rank(method="first", ascending=False)

    total_tokens = df["token_count"].sum()
    df["token_pct"] = df["token_count"] / total_tokens # df["token_count"].apply(lambda x: x / total_tokens)

    df = df.sort_values(by="token_rank")
    df["token_pct_rt"] = df["token_pct"].cumsum()

    # merge document stats

    doc_df = pd.DataFrame(doc_counts, columns=["token", "doc_count"])

    df = doc_df.merge(df, on="token")

    total_docs = len(token_sets)
    df["doc_pct"] = df["doc_count"] / total_docs # df["doc_count"].apply(lambda x: x / total_docs)

    return df.sort_values(by="token_rank")

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

    #tokens = np.hstack(df["nlp.tokens"].values) # h/t: https://stackoverflow.com/a/11367444/670433
    #sns.countplot(tokens)
    #plt.show()
    #> hmmm this is hanging pretty bad for me, so how about an example instead...
    #sns.countplot(["all", "the", "kings", "men", "ate", "all", "the", "kings", "hens"])
    #plt.show()

    token_sets = df["nlp.tokens"].values.tolist()
    print(token_sets[0])

    summary_table = summarize(token_sets)

    #sns.distplot(summary_table["doc_pct"])
    #plt.show()

    most_frequent_tokens_table = summary_table[summary_table["rank"] <= 20]

    squarify.plot(sizes=most_frequent_tokens_table["token_pct"], label=most_frequent_tokens_table["token"], alpha=0.8 )
    plt.axis("off")
    plt.show()

    #
    # PROCESSING > VECTORIZING
    #

    # TODO: add column called "nlp.vectors" and add vectorized tokens there
