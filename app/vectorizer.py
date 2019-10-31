
import os

#import spacy

BBC_DOCS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "bbc_docs")

def parse_text_files(dirpath):
    """
    Parses the contents of all text files in a given directory and stores them in memory.
    Param: dirpath (str): path to a directory of .txt files
    Return: list of strings
    """
    texts = []
    filenames = os.listdir(dirpath) #> [... '284.txt']
    txt_filenames = sorted([fn for fn in filenames if fn.endswith(".txt")]) #> ['001.txt' ... '401.txt']
    for txt_filename in txt_filenames:
        txt_filepath = os.path.join(dirpath, txt_filename)
        with open(txt_filepath, "rb") as txt_file:
            texts.append(txt_file.read())
    return texts

if __name__ == "__main__":

    #nlp = spacy.load("en_core_web_lg")

    texts = parse_text_files(BBC_DOCS_DIRPATH)
    print("TEXTS", len(texts))
    print(texts[0])
