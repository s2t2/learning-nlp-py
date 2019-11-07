
import os
import pytest

MY_PREAMBLE = "Friends, Romans, countrymen, lend me your ears; 911"

MY_MESSAGE = " Oh HeY there - so whatr'u  up to later???? \n   Statue of Liberty trip later. \n Text me (123) 456-7890. k cool! "

DOCUMENTS = [
    "all the kings men",
    "ate all the kings hens",
    "until they all got tired and went to sleep zzz"
]
TOKEN_SETS = [doc.split() for doc in DOCUMENTS]
#> [
#    ["all", "the", "kings", "men"],
#    ["ate", "all", "the", "kings", "hens"],
#    ["until", "they", "all", "got", "tired", "and", "went", "to", "sleep", "zzz"]
#]

NOVELS_DIRPATH = os.path.join(os.path.dirname(__file__), "test", "data", "novels")

# prevents unnecessary or duplicative language model loading
# fixture only loaded when a specific test needs it
# module-level fixture scope only invoked once for all tests in the same file
# session-level fixture scope only invoked once for all tests in the suite
@pytest.fixture(scope="session")
def nlp():
    import spacy
    print("LOADING THE LANGUAGE MODEL...")
    return spacy.load("en_core_web_md") #> <class 'spacy.lang.en.English'>
