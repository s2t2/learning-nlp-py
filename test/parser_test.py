from pandas import DataFrame

from app.parser import tokenize, summarize
from conftest import MY_PREAMBLE, MY_MESSAGE, TOKEN_SETS

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

    assert tokenize("Don't do Full-Time") == ["dont", "do", "fulltime"]

def test_summarize():
    df = summarize(TOKEN_SETS)

    print(df.head())
    assert isinstance(df, DataFrame)
