
import pytest

# prevents unnecessary or duplicative language model loading
# fixture only loaded when a specific test needs it
# module-level fixture only invoked once for all tests
@pytest.fixture(scope="module")
def nlp():
    import spacy
    print("LOADING THE LANGUAGE MODEL...")
    return spacy.load("en_core_web_md")
