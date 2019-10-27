

from app.parser import tokenize

MY_PREAMBLE = "Friends, Romans, countrymen, lend me your ears; 911"
MY_MESSAGE = " Oh HeY there - so whatr'u  up to later???? \n   Statue of Liberty trip later. \n Text me (123) 456-7890. k cool! "

def test_tokenize():

    assert tokenize(MY_PREAMBLE) == [
        "friends", "romans", "countrymen",
        "lend", "me", "your", "ears", "911"
    ]

    assert tokenize(MY_MESSAGE) == [
        'oh', 'hey', 'there', 'so', 'whatru', 'up', 'to', 'later',
        'statue', 'of', 'liberty', 'trip', 'later',
        'text', 'me', '123', '4567890', 'k', 'cool'
    ]
