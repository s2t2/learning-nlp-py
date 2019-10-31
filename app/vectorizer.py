
import os
from pprint import pprint

import pandas as pd
#import spacy
from sklearn.feature_extraction.text import CountVectorizer

BBC_DOCS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "bbc_docs")

def parse_text_files(dirpath):
    """
    Parses the contents of all text files in a given directory and stores them in memory.
    Param: dirpath (str): path to a directory of .txt files
    Return: list of dictionaries containing a mapping of text files with their contents
    """
    texts = []
    filenames = os.listdir(dirpath) #> [... '284.txt']
    txt_filenames = sorted([fn for fn in filenames if fn.endswith(".txt")]) #> ['001.txt' ... '401.txt']
    for txt_filename in txt_filenames:
        txt_filepath = os.path.join(dirpath, txt_filename)
        with open(txt_filepath, "rb") as txt_file:
            #texts.append(txt_file.read())
            texts.append({"filename": txt_filename, "text": txt_file.read()})
    return texts

def text_files_dataframe(dirpath):
    """
    Parses the contents of all text files in a given directory and stores them in a dataframe for further use.
    Param: dirpath (str): path to a directory of .txt files
    Return: (pandas.DataFrame)
    """
    text_file_mappings = parse_text_files(dirpath)
    df = pd.DataFrame(text_file_mappings)
    return df

if __name__ == "__main__":

    texts_df = text_files_dataframe(BBC_DOCS_DIRPATH)
    print("TEXTS DATAFRAME", texts_df.shape)
    print(texts_df.head(3))

    cv = CountVectorizer()
    feature_matrix = cv.fit_transform(texts_df["text"]) #> <class 'scipy.sparse.csr.csr_matrix'>
    feature_names = cv.get_feature_names()
    features_df = pd.DataFrame(data=feature_matrix.toarray(), index=texts_df["filename"], columns=feature_names)
    print("FEATURE MATRIX DATAFRAME", features_df.shape)
    print(features_df.head(3))

    df = pd.merge(texts_df, features_df, on="filename")
    print("DATAFRAME WITH TEXTS AND FEATURES")
    first_row = df.iloc[0].to_dict()
    #print(first_row)
    first_row_abbrev = { k: first_row[k] for k in ["filename", "text_x", "ink", "drive", "democracy", "europe"] }
    pprint(first_row_abbrev)
