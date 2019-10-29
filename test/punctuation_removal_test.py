
import string

from conftest import MY_PREAMBLE, MY_MESSAGE

def test_punctuation_removal():
    assert string.punctuation == '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    assert "!" in string.punctuation

    nonpunctuation_characters = [char for char in "Hello World!!!" if char not in string.punctuation]
    assert "".join(nonpunctuation_characters) == "Hello World"

def test_translation_table():

    translation_table = str.maketrans("", "", string.punctuation)
    assert translation_table == {33: None, 34: None, 35: None, 36: None, 37: None, 38: None, 39: None, 40: None, 41: None, 42: None, 43: None, 44: None, 45: None, 46: None, 47: None, 58: None, 59: None, 60: None, 61: None, 62: None, 63: None, 64: None, 91: None, 92: None, 93: None, 94: None, 95: None, 96: None, 123: None, 124: None, 125: None, 126: None}

    assert "Hello World!!!".translate(translation_table) == 'Hello World'

    assert MY_PREAMBLE.translate(translation_table) == 'Friends Romans countrymen lend me your ears 911'

    assert MY_MESSAGE.translate(translation_table) == ' Oh HeY there  so whatru  up to later \n   Statue of Liberty trip later \n Text me 123 4567890 k cool '
