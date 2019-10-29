
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

WORDS = ["is", "are", "be", "was", "ran", "running", "from", "wolves"]
DOCUMENT = " ".join(WORDS) #> 'is are be was ran running from wolves'

def test_nltk_porter_stemmer():
    ps = PorterStemmer()
    assert [ps.stem(word) for word in WORDS] == ["is", "are", "be", "wa", "ran", "run", "from", "wolv"]

def test_nltk_lancaster_stemmer():
    ls = LancasterStemmer()
    assert [ls.stem(word) for word in WORDS] == ["is", "ar", "be", "was", "ran", "run", "from", "wolv"]

def test_nltk_lemmatizer():
    lm = WordNetLemmatizer()
    assert [lm.lemmatize(word) for word in WORDS] == ["is", "are", "be", "wa", "ran", "running", "from", "wolf"]

def test_spacy_lemmatizer(nlp):
    doc = nlp(DOCUMENT) #> <class 'spacy.tokens.doc.Doc'>
    assert [word.lemma_ for word in doc] == ["be", "be", "be", "be", "run", "run", "from", "wolf"]
