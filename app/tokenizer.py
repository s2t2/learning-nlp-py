import os
#from pprint import pprint
from collections import Counter

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from spacy.tokenizer import Tokenizer

import matplotlib.pyplot as plt
import seaborn as sns
import squarify

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEWS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "amazon_reviews.csv")

ALPHANUMERIC_PATTERN = r'[^a-zA-Z ^0-9]' # same as "[^a-zA-Z ^0-9]"

NLP = spacy.load("en_core_web_md")

def tokenize(doc):
    """
    Params: doc (str) the document to tokenize
    Returns: a list of tokens
    """
    #print("TOKENIZING...")
    doc = doc.lower() # normalize case
    doc = re.sub(ALPHANUMERIC_PATTERN, "", doc) # keep only alphanumeric characters
    tokens = doc.split()
    # todo: consider removing stopwords!
    # todo: consider stemming / lemmatizing!
    return tokens

def tokenize_v2(doc):
    """
    Params: doc (str) the document to tokenize
    Returns: a list of tokens
    """
    doc = doc.lower() # normalize case
    doc = re.sub(ALPHANUMERIC_PATTERN, "", doc) # keep only alphanumeric characters
    tokens = doc.split()
    stop_words = stopwords.words("english")
    tokens = [token for token in tokens if not token in stop_words]
    # todo: consider stemming / lemmatizing!
    return tokens

def tokenize_v22(doc):
    """
    Params: doc (str) the document to tokenize
    Returns: a list of tokens
    """
    doc = doc.lower() # normalize case
    doc = re.sub(ALPHANUMERIC_PATTERN, "", doc) # keep only alphanumeric characters
    tokens = doc.split()
    stop_words = stopwords.words("english")
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens if not token in stop_words]
    return tokens

def tokenize_v3(my_doc, my_nlp=NLP):
    """
    Params:
        my_doc (str) the document to tokenize
        my_nlp (spacy.lang.en.English) one of spacy's natural language models
    Returns: a list of tokens
    """
    doc = my_nlp(my_doc) #> <class 'spacy.tokens.doc.Doc'>
    tokens = [token.text.lower() for token in doc if
        token.is_stop == False
        and token.is_punct == False
        and token.is_space == False
    ]
    # todo: consider stemming / lemmatizing!
    return tokens

def tokenize_v4(my_doc, my_nlp=NLP):
    """
    Params:
        my_doc (str) the document to tokenize
        my_nlp (spacy.lang.en.English) one of spacy's natural language models
    Returns: a list of tokens
    """
    doc = my_nlp(my_doc) #> <class 'spacy.tokens.doc.Doc'>
    tokens = [token.lemma_.lower() for token in doc if token.is_stop == False and token.is_punct == False and token.is_space == False]
    return tokens

def tokenize_v5(my_docs, my_nlp=NLP, batch_size=200):
    """
    Uses a tokenizer pipeline for performance gains (JK still very slow).
    Params:
        my_docs (list of str, or dataframe column of str) the documents to tokenize
        my_nlp (spacy.lang.en.English) one of spacy's natural language models
    Returns: a token set (list of token lists)
    """
    #print("TOKENIZING (v5)...")
    tokenizer = Tokenizer(my_nlp.vocab)
    token_sets = []
    for doc in tokenizer.pipe(my_docs, batch_size=batch_size):
        # tokens = [token.lemma_.lower() for token in doc if token.is_stop == False and token.is_punct == False and token.is_space == False]
        # ... for some reason there are special characters, so maybe...
        clean_text = re.sub(ALPHANUMERIC_PATTERN, "", doc.text)
        clean_doc = my_nlp(clean_text)
        tokens = [token.lemma_.lower() for token in clean_doc if token.is_stop == False and token.is_punct == False and token.is_space == False]
        # ... hmm stopwords are still making their way through if the lemma is a stopword
        token_sets.append(tokens)
    return token_sets

def summarize(token_sets):
    """
    Param: token_sets a list of token lists
    """

    print("COMPILING TOKEN SUMMARY TABLE...")
    token_counter = Counter()
    doc_counter = Counter()

    for tokens in token_sets:
        token_counter.update(tokens)
        doc_counter.update(set(tokens)) # removes duplicate tokens so they only get counted once per doc!

    token_counts = zip(token_counter.keys(), token_counter.values())
    doc_counts = zip(doc_counter.keys(), doc_counter.values())

    token_df = pd.DataFrame(token_counts, columns=["token", "count"])
    doc_df = pd.DataFrame(doc_counts, columns=["token", "doc_count"])

    df = doc_df.merge(token_df, on="token")
    total_tokens = df["count"].sum()
    total_docs = len(token_sets)

    df["rank"] = df["count"].rank(method="first", ascending=False)

    df["pct"] = df["count"] / total_tokens # df["token_count"].apply(lambda x: x / total_tokens)

    df = df.sort_values(by="rank")
    df["running_pct"] = df["pct"].cumsum()

    df["doc_pct"] = df["doc_count"] / total_docs # df["doc_count"].apply(lambda x: x / total_docs)

    ordered_columns = ["token", "rank", "count", "pct", "running_pct", "doc_count", "doc_pct"]
    return df.reindex(columns=ordered_columns).sort_values(by="rank")

def plot_top_tokens(token_sets):
    summary_table = summarize(token_sets)
    top_tokens_table = summary_table[summary_table["rank"] <= 20]
    print("PLOTTING TOP TOKENS...")

    #sns.distplot(summary_table["pct"])
    #plt.show()

    squarify.plot(sizes=top_tokens_table["pct"], label=top_tokens_table["token"], alpha=0.8)
    plt.axis("off")
    plt.show()

def plot_tokens(tokens=["all", "the", "kings", "men", "ate", "all", "the", "kings", "hens"]):
    #tokens = np.hstack(df["nlp.tokens"].values) # h/t: https://stackoverflow.com/a/11367444/670433
    #sns.countplot(tokens)
    #plt.show()
    #> hmmm this is hanging pretty bad for me, so how about an example instead...
    sns.countplot(tokens)
    plt.show()

if __name__ == "__main__":

    #
    # LOADING
    #

    df = pd.read_csv(REVIEWS_CSV_FILEPATH)

    print(df[["reviews.rating", "reviews.text"]])
    print("REVIEW RATINGS...")
    print(df["reviews.rating"].value_counts())
    #df["reviews.rating"].value_counts().sort_index().plot.bar()
    #plt.show()
    print("REVIEW TEXT...")
    print(df["reviews.text"].str.len().value_counts())
    #df["reviews.text"].str.len().plot.hist()
    #plt.show()

    #
    # TOKENIZING
    #

    df["nlp.tokens"] = df["reviews.text"].apply(lambda txt: tokenize(txt))
    print(df["nlp.tokens"].head())
    plot_top_tokens(df["nlp.tokens"].values.tolist())

    df["nlp.tokens.v2"] = df["reviews.text"].apply(lambda txt: tokenize_v2(txt))
    print(df["nlp.tokens.v2"].head())
    plot_top_tokens(df["nlp.tokens.v2"].values.tolist())

    #df["nlp.tokens.v22"] = df["reviews.text"].apply(lambda txt: tokenize_v22(txt))
    #print(df["nlp.tokens.v22"].head())
    #plot_top_tokens(df["nlp.tokens.v22"].values.tolist())

    #nlp = spacy.load("en_core_web_md")

    #df["nlp.tokens.v3"] = df["reviews.text"].apply(lambda txt: tokenize_v3(txt, nlp))
    #print(df["nlp.tokens.v3"].head())
    #plot_top_tokens(df["nlp.tokens.v3"].values.tolist())

    #df["nlp.tokens.v4"] = df["reviews.text"].apply(lambda txt: tokenize_v4(txt, nlp))
    #print(df["nlp.tokens.v4"].head())
    #plot_top_tokens(df["nlp.tokens.v4"].values.tolist())

    #df["nlp.tokens.v5"] = tokenize_v5(df["reviews.text"], nlp) # OR tokenize_v5(df["reviews.text"].values.tolist(), nlp)
    #print(df["nlp.tokens.v5"].head())
    #plot_top_tokens(df["nlp.tokens.v5"].values.tolist())

    #
    # VECTORIZING
    #

    # TODO: add column called "nlp.vectors" and add vectorized tokens there
