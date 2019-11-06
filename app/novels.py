
import os
from pprint import pprint

import numpy as np
import pandas as pd
import re

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS

ALPHANUMERIC_PATTERN = r'[^a-zA-Z ^0-9]'

NOVELS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "novels")

def tokenize(txt):
    #tokens = txt.strip("\n").lower().split()
    txt = re.sub(ALPHANUMERIC_PATTERN, "", txt)
    tokens = txt.lower().split()
    #bow = dictionary.doc2bow(tokens)
    return tokens

# In practice, corpora may be very large, so loading them into memory may be impossible.
# Gensim intelligently handles such corpora by streaming them one document at a time.
def token_stream(path):
    txt_filenames = [filename for filename in os.listdir(path) if filename.endswith(".txt")]
    for txt_filename in txt_filenames:
        txt_filepath = os.path.join(NOVELS_DIRPATH, txt_filename)
        with open(txt_filepath) as txt_file:
            tokens = tokenize(str(txt_file.read()))
            yield tokens

def parse_topics(lda):
    """lda (gensim.models.ldamulticore.LdaMulticore) a pre-fit LDA model"""
    parsed_response = []
    topics_response = lda.print_topics()
    for topic_row in topics_response:
        topics = topic_row[1] #> '0.067*"sleep" + 0.067*"got" + 0.067*"went" + 0.067*"until" + 0.067*"to" + 0.067*"tired" + 0.067*"they" + 0.067*"all" + 0.067*"ate" + 0.067*"the"'
        topic_pairs = [s.replace('"', "").split("*") for s in topics.split(" + ")] #> [ ['0.067', 'sleep'], ['0.067', 'got'], [], etc... ]
        doc_topics = {}
        for topic_pair in topic_pairs:
            doc_topics[topic_pair[1]] = float(topic_pair[0])
        #print(doc_topics) #> {'sleep': 0.067, 'got': 0.067, etc}
        parsed_response.append(doc_topics)
    return parsed_response

if __name__ == "__main__":

    dictionary = Dictionary(token_stream(NOVELS_DIRPATH))
    dictionary.filter_extremes(no_below=10, no_above=0.66) # excludes terms like "the", "to", "and", "of", "i", etc.
    print("-------------")
    print("TOKENS", len(dictionary.token2id), list(dictionary.token2id.items())[0:4], "...")

    bags_of_words = [dictionary.doc2bow(tokens) for tokens in token_stream(NOVELS_DIRPATH)]
    print("-------------")
    print("BAGS OF WORDS (CORPUS)", len(bags_of_words), bags_of_words[0])

    lda = LdaMulticore(corpus=bags_of_words, id2word=dictionary, random_state=723812, num_topics=15, passes=10, workers=4)
    print("-------------")
    print("LDA MODEL", type(lda))

    results = lda.print_topics()
    print("-------------")
    print("TOPICS (RAW RESULTS)...")
    print(results)

    parsed_topics = parse_topics(lda)
    print("-------------")
    print("TOPICS (PARSED RESULTS)...")
    pprint(parsed_topics)
