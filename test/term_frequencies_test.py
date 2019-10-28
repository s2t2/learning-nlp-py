
from collections import Counter

TOKENS = [
    ["all", "the", "kings", "men"],
    ["ate", "all", "the", "kings", "hens"],
    ["until", "they", "all", "got", "tired", "and", "went", "to", "sleep", "zzz"]
]

def test_counter():

    counter = Counter()

    for token in TOKENS:
        counter.update(token) # pass a list of words in to group by word, pass a word in to group by char

    breakpoint()
    #assert counter.most_common(3) == [('e', 10), ('t', 9), ('l', 8)]
    assert counter.most_common(3) == [('all', 3), ('the', 2), ('kings', 2)]




#from sklearn.feature_extraction.text import CountVectorizer
#
#def test_sklearn_counter():
#    my_dist = FreqDist({'the': 3, 'dog': 2, 'not': 1})
#    print(my_dist.most_common(2))

#from nltk.probability import FreqDist
#
#def test_nltk_counter():
#    my_dist = FreqDist({'the': 3, 'dog': 2, 'not': 1})
#    print(my_dist.most_common(2))
