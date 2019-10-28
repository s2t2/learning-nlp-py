
import pytest

MY_PREAMBLE = "Friends, Romans, countrymen, lend me your ears; 911"
MY_MESSAGE = " Oh HeY there - so whatr'u  up to later???? \n   Statue of Liberty trip later. \n Text me (123) 456-7890. k cool! "

# prevents unnecessary or duplicative language model loading
# fixture only loaded when a specific test needs it
# module-level fixture scope only invoked once for all tests in the same file
# session-level fixture scope only invoked once for all tests in the suite
@pytest.fixture(scope="session")
def nlp():
    import spacy
    print("LOADING THE LANGUAGE MODEL...")
    return spacy.load("en_core_web_md")
