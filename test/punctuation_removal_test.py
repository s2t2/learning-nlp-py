
import string

from conftest import MY_PREAMBLE, MY_MESSAGE

def test_punctuation_removal():

    assert string.punctuation == '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    assert "!" in string.punctuation

    translation_table = str.maketrans("", "", string.punctuation)
    assert translation_table == {33: None, 34: None, 35: None, 36: None, 37: None, 38: None, 39: None, 40: None, 41: None, 42: None, 43: None, 44: None, 45: None, 46: None, 47: None, 58: None, 59: None, 60: None, 61: None, 62: None, 63: None, 64: None, 91: None, 92: None, 93: None, 94: None, 95: None, 96: None, 123: None, 124: None, 125: None, 126: None}

    translated_preamble = MY_PREAMBLE.translate(translation_table)
    assert translated_preamble == 'Friends Romans countrymen lend me your ears 911'

    translated_message = MY_MESSAGE.translate(translation_table)
    assert translated_message == ' Oh HeY there  so whatru  up to later \n   Statue of Liberty trip later \n Text me 123 4567890 k cool '
