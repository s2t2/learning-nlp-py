
import os

import numpy as np
import pandas as pd
#import re

from gensim.corpora import Dictionary
#from gensim.models.ldamulticore import LdaMulticore
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS

NOVELS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "novels")

# In practice, corpora may be very large, so loading them into memory may be impossible.
# Gensim intelligently handles such corpora by streaming them one document at a time.
# See Corpus Streaming – One Document at a Time for details.

def tokenize(txt):
    tokens = txt.strip("\n").lower().split()
    #bow = dictionary.doc2bow(tokens)
    return tokens

def token_stream(path):
    txt_filenames = [filename for filename in os.listdir(path) if filename.endswith(".txt")]
    for txt_filename in txt_filenames:
        txt_filepath = os.path.join(NOVELS_DIRPATH, txt_filename)
        with open(txt_filepath) as txt_file:
            tokens = tokenize(str(txt_file.read()))
            yield tokens

if __name__ == "__main__":
    #
    # corpora.Dictionary
    #   + encapsulates the mapping between normalized words and their integer ids
    #   + defines the vocabulary of all words that our processing knows about

    texts = ["all the kings men", "ate all the kings hens", "until they all got tired and went to sleep zzz"]
    token_sets = [tokenize(text) for text in texts]
    d = Dictionary(token_sets)
    #print(list(d.items()))
    print(d.token2id) #> {'all': 0, 'kings': 1, 'men': 2, 'the': 3, 'ate': 4, 'hens': 5, 'and': 6, 'got': 7, 'sleep': 8, 'they': 9, 'tired': 10, 'to': 11, 'until': 12, 'went': 13, 'zzz': 14}

    new_tokens = ["all", "kings", "queens", "jacks"]
    print(new_tokens)
    print(d.doc2idx(new_tokens))
    print(d.doc2bow(new_tokens))
    #bow_corpus = [d.doc2bow(text) for text in texts]

    #breakpoint()
    #lda = LdaMulticore(corpus=token_sets, id2word=dictionary)
    #print(lda.print_topics())








    dictionary = Dictionary(token_stream(NOVELS_DIRPATH))
    print(dictionary.token2id)

    #dictionary.filter_extremes(no_below=10, no_above=0.75)
    #lda = LdaMulticore(corpus=token_sets, id2word=dictionary, random_state=723812, num_topics=15, passes=10, workers=4)
    #print(lda.print_topics())
