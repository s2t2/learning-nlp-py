
import os
from pprint import pprint

import pandas as pd
#import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

BBC_DOCS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "bbc_docs")
EXPORTS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "exports")

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

def tfidf_vectorized_dataframe(texts_df, dense=False):
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

#def cosine_similarity_matrix(vectorized_df):
#    """
#    Param: vectorized_df (pd.DataFrame) a dataframe with columns "txt.filename" and "txt.contents",
#            ... and also a column for each feature (feature matrix)
#    """
#    features_df = vectorized_df
#    del features_df["txt.filename"]
#    del features_df["txt.contents"]
#    similarity_matrix = cosine_similarity(features_df)
#    return similarity_matrix # pd.DataFrame(similarity_matrix)

def cosine_similarity_dataframe(vectorized_df):
    """
    Param: vectorized_df (pd.DataFrame) a dataframe with columns "txt.filename" and "txt.contents",
            ... and also a column for each feature (feature matrix)
    """
    docs_df = vectorized_df[["txt.filename", "txt.contents"]]

    #features_df = vectorized_df
    #del features_df["txt.filename"]
    #del features_df["txt.contents"]
    features_df = vectorized_df.loc[:, ~vectorized_df.columns.isin(["txt.filename", "txt.contents"])]

    similarity_matrix = cosine_similarity(features_df)
    similarity_df = pd.DataFrame(similarity_matrix)

    #print("COMBINING", docs_df.shape, similarity_df.shape) #> (401, 2) (401, 401)
    combined_df = pd.concat([docs_df, similarity_df], axis=1)
    return combined_df

if __name__ == "__main__":

    texts_df = text_files_dataframe(BBC_DOCS_DIRPATH)
    print("---------------------")
    print("TEXTS DATAFRAME")
    print(texts_df.shape)
    print(texts_df.head(3))

    print("---------------------")
    print("COUNT VECTOR (SPARSE)")
    df = count_vectorized_dataframe(texts_df)
    print(df.shape)
    df.to_csv(os.path.join(EXPORTS_DIRPATH, "counts_matrix.csv"))
    first_row = df.iloc[0].to_dict()
    first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "ink", "drive", "democracy", "europe"] }
    pprint(first_row_abbrev)

    print("---------------------")
    print("TFIDF VECTOR (SPARSE)")
    df = tfidf_vectorized_dataframe(texts_df)
    print(df.shape) #> (401, 12098)
    df.to_csv(os.path.join(EXPORTS_DIRPATH, "tfidf_matrix.csv"))
    first_row = df.iloc[0].to_dict()
    first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "ink", "drive", "democracy", "europe"] }
    pprint(first_row_abbrev)

    #print("---------------------")
    #print("TFIDF VECTOR (DENSE)")
    #df = tfidf_vectorized_dataframe(texts_df, dense=True)
    #first_row = df.iloc[0].to_dict()
    ##first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "txt.contents", "ink", "drive", "democracy", "europe"] }
    #first_row_abbrev = { k: first_row[k] for k in ["txt.filename", "ink", "drive", "democracy", "europe"] }
    #pprint(first_row_abbrev)

    print("---------------------")
    print("DOCUMENT SIMILARITY (COSINE)")
    #similarity_matrix = cosine_similarity_matrix(df)
    #print(type(similarity_matrix), similarity_matrix.shape)
    #print(sorted(similarity_matrix[0])[0:10])
    similarity_df = cosine_similarity_dataframe(df)
    similarity_df.to_csv(os.path.join(EXPORTS_DIRPATH, "tfidf_cosine_similarities.csv"))

    first_doc = similarity_df.iloc[0]
    print("FIRST DOC")
    print(first_doc)
    similar_docs = similarity_df.loc[:, ~similarity_df.columns.isin(["txt.filename", "txt.contents"])].iloc[0]
    most_similar_docs = similar_docs.sort_values(ascending=False)[0:10]
    print("SIMILAR DOCS")
    print(most_similar_docs)
    #breakpoint()

    print("---------------------")
    print("DOCUMENT SIMILARITY (KNN)")

    model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree") # algorithm="kd_tree", etc.
    print("MODEL", model)
    dtm = df.loc[:, ~df.columns.isin(["txt.filename", "txt.contents"])]
    model.fit(dtm)

    results = model.kneighbors([dtm.iloc[0]])
    print("RESULTS", results)
    print("DISTANCES", results[0])
    print("DOCUMENTS", results[1])

    for doc_id in results[1][0]:
        print("-----")
        print("DOC", doc_id)
        print(df.iloc[doc_id]["txt.contents"][0:200])
