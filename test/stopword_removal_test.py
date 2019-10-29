
from nltk.corpus import stopwords # FYI: need to run nltk.download() or nltk.download('stopwords') on your machine for this to work

def test_nltk_stopwords(nlp):
    stop_words = stopwords.words("english")

    assert stop_words == [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', "don't",
        'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
        'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ]
    assert "the" in stop_words
    assert "pajamas" not in stop_words

def test_spacy_stopwords(nlp):
    stop_words = nlp.Defaults.stop_words

    assert stop_words == {
        'off', 'alone', 'hereupon', 'well', 'empty', '‘m', 'get', 'otherwise', 'last', 'himself',
        'most', 'all', 'ten', 'anything', 'these', 'fifty', 'hereafter', 'will', 'ever', 'besides',
        'hers', 'me', 'side', 'except', 'only', 'n’t', 'anyone', 'never', 'than', "'m", 'because',
        'nor', 'yourself', 'yourselves', 'namely', 'regarding', 'yet', 'wherein', 'many', 'she', 'this',
        '‘re', 'see', 'it', 'too', 'perhaps', 'became', 'out', 'being', 'is', 'whose', 'n‘t', 'via', 'who',
        'sometimes', 'two', 'why', 'under', 'neither', 'therefore', 'using', 'ourselves', 'how', 'throughout',
        'around', 'eight', 'before', 'since', 'were', 'toward', 'whereafter', 'should', 'less', 'bottom',
        'eleven', 'whence', 'say', 'his', 'of', "'ll", 'are', 'per', 'six', 'their', 'which', 'might',
        "'ve", 'thence', 'can', 'almost', 'between', 'whom', 'everyone', 'forty', 'herself', 'as', 'done',
        'has', 'without', 'seems', 'everything', 'been', 'seem', 'latter', 'not', 'always', 'fifteen', 'to',
        'however', 'him', 'be', 'least', 'much', 'others', 'until', 'made', 'go', 'mostly', 'hence', 'by',
        'themselves', 'our', 'another', 'full', 'no', 'quite', 'what', 'you', 'seeming', 'here', 'becoming',
        'just', 'beyond', 'whoever', 'would', 'we', 'show', '’m', 'after', 'already', 'together', 'when',
        'move', 'put', 'had', 'through', 'due', 'or', 'may', '‘ve', 'twelve', 'twenty', 'onto', 'elsewhere',
        '’s', 'its', 'amount', 'every', 'whither', 'name', 'yours', 'someone', 'therein', 'enough', 'four',
        'seemed', 'though', 'sometime', 'none', '‘ll', 'but', 'over', 'keep', 'thereby', 'they', 'during',
        'latterly', 'those', '‘d', 'he', 'unless', 'although', 'indeed', 'both', 'was', 'along', 'beside',
        'herein', 'up', 'have', 'somewhere', 'upon', 'ca', 'into', 'while', 'sixty', 'cannot', '’ve', 'other',
        'there', 'used', 'across', 'become', 'ours', 'few', 'part', 'such', 'formerly', 'three', 'an',
        'becomes', '’ll', 'myself', 'on', "'d", 'amongst', 'either', 'hundred', 'same', '‘s', 'nevertheless',
        'now', 'more', 'behind', 'first', 'often', 'five', 'afterwards', 'whereupon', 'so', 'beforehand',
        'very', 'own', 'did', 'further', 'at', 'below', 'nowhere', 'nothing', 'anyhow', "'re", 'also', 'in',
        'among', 'where', 'am', 'serious', 'from', 'does', 'former', 'everywhere', 'towards', 'whatever',
        'and', 'next', 'then', "n't", 'thereafter', 'even', 're', 'back', 'moreover', 'with', 'some', 'each',
        'within', 'thru', 'itself', 'one', 'nobody', 'nine', 'them', 'hereby', 'take', 'meanwhile', 'whereby',
        'against', 'whenever', 'whether', 'various', 'above', 'for', 'that', 'somehow', 'whereas', 'your',
        'make', 'wherever', 'front', 'about', 'my', 'several', "'s", 'do', 'once', 'give', 'i', 'something',
        'else', 'still', 'doing', 'a', 'please', 'anywhere', 'really', 'mine', 'if', 'thus', 'us', 'again',
        'the', 'must', 'top', 'could', 'anyway', 'third', '’d', 'thereupon', 'rather', 'whole', 'any', 'call',
        '’re', 'down', 'noone', 'her'
    }
    assert "the" in stop_words
    assert "pajamas" not in stop_words

    custom_stop_words = nlp.Defaults.stop_words.union(["@POTUS", "pajamas"])
    assert "pajamas" in custom_stop_words
