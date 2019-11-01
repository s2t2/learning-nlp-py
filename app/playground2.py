

import os

import pandas as pd

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
BBC_DOCS_DIRPATH = os.path.join(DATA_DIRPATH, "bbc_docs")

if __name__ == "__main__":

    print("PARSING TEXT FILES...")

    filenames = os.listdir(BBC_DOCS_DIRPATH) #> [... '284.txt']
    txt_filenames = sorted([fn for fn in filenames if fn.endswith(".txt")]) #> ['001.txt' ... '401.txt']
    for txt_filename in txt_filenames[0:4]:
        txt_filepath = os.path.join(BBC_DOCS_DIRPATH, txt_filename)
        with open(txt_filepath, "rb") as txt_file:
            print("-----")
            print(txt_filename)
            print(txt_file.read()[0:200])

    breakpoint()
