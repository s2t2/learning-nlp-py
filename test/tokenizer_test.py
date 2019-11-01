from pandas import DataFrame

from app.tokenizer import tokenize, tokenize_v2, tokenize_v22, tokenize_v3, tokenize_v4, tokenize_v5, summarize
from conftest import MY_PREAMBLE, MY_MESSAGE, TOKEN_SETS, DOCUMENTS

def test_tokenize():
    assert tokenize("Don't do Full-Time") == ["dont", "do", "fulltime"]
    assert tokenize(MY_PREAMBLE) == ["friends", "romans", "countrymen", "lend", "me", "your", "ears", "911"]
    assert tokenize(MY_MESSAGE) == [
        'oh', 'hey', 'there', 'so', 'whatru', 'up', 'to', 'later',
        'statue', 'of', 'liberty', 'trip', 'later',
        'text', 'me', '123', '4567890', 'k', 'cool'
    ]

def test_tokenize_v2():
    assert tokenize_v2("Don't do Full-Time") == ['dont', 'fulltime']
    assert tokenize_v2(MY_PREAMBLE) == ['friends', 'romans', 'countrymen', 'lend', 'ears', '911']
    assert tokenize_v2(MY_MESSAGE) == [
        'oh', 'hey', 'whatru', 'later',
        'statue', 'liberty', 'trip', 'later',
        'text', '123', '4567890', 'k', 'cool'
    ]

def test_tokenize_v22():
    assert tokenize_v22("Don't do Full-Time") == ['dont', 'fulltim']
    assert tokenize_v22(MY_PREAMBLE) == ['friend', 'roman', 'countrymen', 'lend', 'ear', '911']
    assert tokenize_v22(MY_MESSAGE) == [
        'oh', 'hey', 'whatru', 'later',
        'statu', 'liberti', 'trip', 'later',
        'text', '123', '4567890', 'k', 'cool'
    ]

def test_tokenize_v3(nlp):
    assert tokenize_v3("Don't do Full-Time", nlp) == ['time']
    assert tokenize_v3(MY_PREAMBLE, nlp) == ['friends', 'romans', 'countrymen', 'lend', 'ears', '911']
    assert tokenize_v3(MY_MESSAGE, nlp) == [
        'oh', 'hey', "whatr'u", 'later',
        'statue', 'liberty', 'trip', 'later',
        'text', '123', '456', '7890', 'k', 'cool'
    ]

def test_tokenize_v4(nlp):
    assert tokenize_v4("Don't do Full-Time", nlp) == ['time']
    assert tokenize_v4(MY_PREAMBLE, nlp) == ['friend', 'romans', 'countryman', 'lend', 'ear', '911']
    assert tokenize_v4(MY_MESSAGE, nlp) == [
        'oh', 'hey', "whatr'u", 'later',
        'statue', 'liberty', 'trip', 'later',
        'text', '123', '456', '7890', 'k', 'cool'
    ]

def test_tokenize_v5(nlp):
    assert tokenize_v5(["Don't do Full-Time"], nlp) == [['not', 'fulltime']]
    assert tokenize_v5([MY_PREAMBLE], nlp) == [['friend', 'romans', 'countryman', 'lend', 'ear', '911']]
    assert tokenize_v5([MY_MESSAGE], nlp) == [[
        'oh', 'hey', 'whatru', 'later',
        'statue', 'liberty', 'trip', 'later',
        'text', '123', '4567890', 'k', 'cool'
    ]]
    assert tokenize_v5(DOCUMENTS, nlp) == [
        ['king', 'man'],
        ['eat', 'king', 'hen'],
        ['get', 'tired', 'go', 'sleep', 'zzz']
    ]

def test_summarize():
    df = summarize(TOKEN_SETS)
    print(df.head())

    assert isinstance(df, DataFrame)

    assert df.columns.tolist() == [
        'token', 'rank', 'count', 'pct', 'running_pct', 'doc_count', 'doc_pct'
    ]

    assert df.to_dict("records")[0] == {
        'token': 'all',
        'rank': 1.0,
        'count': 3,
        'pct': 0.15789473684210525,
        'running_pct': 0.15789473684210525,
        'doc_count': 3,
        'doc_pct': 1.0
    }
