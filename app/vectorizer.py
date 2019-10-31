
import os
from pprint import pprint

import pandas as pd
#import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
            texts.append({"txt.filename": txt_filename, "txt.contents": txt_file.read()}) # using "txt." prefixes here because later when this df is merged with the features df, if any of the feature column names are "text" for example, it will change the column names to "text_x" vs "text_y", so just namespace and lessen the chance of that happening...
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

def count_vectorized_dataframe(texts_df):
    """
    Param: texts_df (pd.DataFrame) a dataframe with columns "txt.filename" and "txt.contents"
    """
    cv = CountVectorizer()
    feature_matrix = cv.fit_transform(texts_df["txt.contents"]) #> <class 'scipy.sparse.csr.csr_matrix'>
    data = feature_matrix.toarray()
    feature_names = cv.get_feature_names()
    features_df = pd.DataFrame(data=data, index=texts_df["txt.filename"], columns=feature_names)
    return pd.merge(texts_df, features_df, on="txt.filename")

def tfidf_vectorized_dataframe(texts_df, dense=True):
    """
    Param: texts_df (pd.DataFrame) a dataframe with columns "txt.filename" and "txt.contents"
    """
    tv = TfidfVectorizer()
    feature_matrix = tv.fit_transform(texts_df["txt.contents"]) #> <class 'scipy.sparse.csr.csr_matrix'>
    if dense == True:
        data = feature_matrix.todense()
    else:
        data = feature_matrix.toarray()
    feature_names = tv.get_feature_names()
    features_df = pd.DataFrame(data=data, index=texts_df["txt.filename"], columns=feature_names)
    return pd.merge(texts_df, features_df, on="txt.filename")

if __name__ == "__main__":

    texts_df = text_files_dataframe(BBC_DOCS_DIRPATH)
    print("---------------------")
    print("TEXTS DATAFRAME", texts_df.shape)
    print(texts_df.head(3))

    print("---------------------")
    print("COUNT VECTOR (SPARSE)")
    df = count_vectorized_dataframe(texts_df)
    first_row = df.iloc[0].to_dict()
    #first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "txt.contents", "ink", "drive", "democracy", "europe"] }
    first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "ink", "drive", "democracy", "europe"] }
    pprint(first_row_abbrev)

    print("---------------------")
    print("TFIDF VECTOR (SPARSE)")
    df = tfidf_vectorized_dataframe(texts_df)
    first_row = df.iloc[0].to_dict()
    #first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "txt.contents", "ink", "drive", "democracy", "europe"] }
    first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "ink", "drive", "democracy", "europe"] }
    pprint(first_row_abbrev)
