

import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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
    print("FEATURES", len(cv.get_feature_names()))
    #print(cv.get_feature_names())
    print("MATRIX")
    print(matrix.toarray())
    print(matrix.toarray()[0].tolist())

    # GOAL: construct frequency matrix (TF-IDF)
    tv = TfidfVectorizer()
    matrix = tv.fit_transform(documents) #> <class 'scipy.sparse.csr.csr_matrix'>
    print("FEATURES", len(tv.get_feature_names()))
    #print(cv.get_feature_names())
    print("MATRIX")
    print(matrix.toarray())
    print(matrix.toarray()[0].tolist())

    # GOAL: identify which documents / articles are most similar to the first

    similarity_matrix = cosine_similarity(matrix.toarray())
    similarity_df = pd.DataFrame(similarity_matrix)
    first_doc = similarity_df.iloc[0]
    print(first_doc)
    similar_docs = first_doc.sort_values(ascending=False)[0:10]
    print(documents[0])
    print(documents[332])
    print(documents[36])

    # GOAL: identify which documents / articles are most similar to the first

    dtm = pd.DataFrame(matrix.toarray())

    model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree") # algorithm="kd_tree", etc.
    print("MODEL", model)

    model.fit(dtm)

    results = model.kneighbors([dtm.iloc[0]])

    print("RESULTS", results)
    print("DISTANCES", results[0])
    print("DOCUMENTS", results[1])

    for doc_id in results[1][0]:
        print("-----")
        print("DOC", doc_id)
        print(documents[doc_id][0:200])
