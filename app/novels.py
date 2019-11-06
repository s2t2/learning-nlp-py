
import os

import numpy as np
import pandas as pd
import re

#import gensim
from gensim.corpora import Dictionary # a mapping between words and their integer ids
#from gensim.models.ldamulticore import LdaMulticore
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS

#NOVELS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "novels")

# In practice, corpora may be very large, so loading them into memory may be impossible.
# Gensim intelligently handles such corpora by streaming them one document at a time.
# See Corpus Streaming – One Document at a Time for details.

def tokenize(txt):
    tokens = txt.strip("\n").lower().split()
    #bow = dictionary.doc2bow(tokens)
    return tokens

#def token_stream(path):
#    txt_filenames = [filename for filename in os.listdir(path) if filename.ends_with(".txt")]
#    for txt_filename in txt_filenames:
#        with open(txt_filename) as txt_file:
#            tokens = tokenize(str(txt_file.read()))
#            yield tokens

if __name__ == "__main__":
    texts = ["all the kings men", "ate all the kings hens", "until they all got tired and went to sleep zzz"]
    token_sets = [tokenize(text) for text in texts]
    d = Dictionary(token_sets)
    #print(list(d.keys()))
    #print(list(d.values()))
    print(list(d.items()))

    #breakpoint()

    #exit()
    #dictionary = Dictionary(token_stream(path))
    ##dictionary.filter_extremes(no_below=10, no_above=0.75)
    #lda = LdaMulticore(corpus=token_sets, id2word=dictionary, random_state=723812, num_topics=15, passes=10, workers=4)
    #print(lda.print_topics())
