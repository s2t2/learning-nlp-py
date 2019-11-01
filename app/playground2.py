

import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
BBC_DOCS_DIRPATH = os.path.join(DATA_DIRPATH, "bbc_docs")

if __name__ == "__main__":

    print("PARSING TEXT FILES...")

    documents = []

    filenames = os.listdir(BBC_DOCS_DIRPATH) #> [... '284.txt']
    txt_filenames = sorted([fn for fn in filenames if fn.endswith(".txt")]) #> ['001.txt' ... '401.txt']
    for txt_filename in txt_filenames:
        txt_filepath = os.path.join(BBC_DOCS_DIRPATH, txt_filename)
        with open(txt_filepath, "rb") as txt_file:
            #print("-----")
            #print(txt_filename)
            #print(txt_file.read()[0:200])
            #print(txt_file.read())
            documents.append(txt_file.read())

    print(f"PARSED {len(documents)} DOCUMENTS..")

    # GOAL: construct frequency matrix
    cv = CountVectorizer()
    #documents = ["doc 1 contents", "doc 2 contents", "doc 3 contents"]
    matrix = cv.fit_transform(documents) #> <class 'scipy.sparse.csr.csr_matrix'>
    #print("FEATURES")
    #print(cv.get_feature_names())
    print("MATRIX")
    print(matrix.toarray())
    print(matrix.toarray()[0].tolist())

    breakpoint()
