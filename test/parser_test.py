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

    assert sorted(df.columns.tolist()) == [
        'doc_count', 'doc_pct', 'token', 'token_count', 'token_pct', 'token_pct_rt', 'token_rank'
    ]

    records = df.to_dict("records")[0] == {
        'token': 'all',
        'doc_count': 3,
        'token_count': 3,
        'token_rank': 1.0,
        'token_pct': 0.15789473684210525,
        'token_pct_rt': 0.15789473684210525,
        'doc_pct': 1.0
    }

    breakpoint()

    #assert df.to_dict("records") == [
    #    {'token': 'all', 'doc_count': 3, 'token_count': 3, 'token_rank': 1.0, 'token_pct': 0.15789473684210525, 'token_pct_rt': 0.15789473684210525, 'doc_pct': 1.0},
    #    {'token': 'the', 'doc_count': 2, 'token_count': 2, 'token_rank': 2.0, 'token_pct': 0.10526315789473684, 'token_pct_rt': 0.2631578947368421, 'doc_pct': 0.6666666666666666},
    #    {'token': 'kings', 'doc_count': 2, 'token_count': 2, 'token_rank': 3.0, 'token_pct': 0.10526315789473684, 'token_pct_rt': 0.3684210526315789, 'doc_pct': 0.6666666666666666},
    #    {'token': 'men', 'doc_count': 1, 'token_count': 1, 'token_rank': 4.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.42105263157894735, 'doc_pct': 0.3333333333333333},
    #    {'token': 'ate', 'doc_count': 1, 'token_count': 1, 'token_rank': 5.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.47368421052631576, 'doc_pct': 0.3333333333333333},
    #    {'token': 'hens', 'doc_count': 1, 'token_count': 1, 'token_rank': 6.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.5263157894736842, 'doc_pct': 0.3333333333333333},
    #    {'token': 'until', 'doc_count': 1, 'token_count': 1, 'token_rank': 7.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.5789473684210527, 'doc_pct': 0.3333333333333333},
    #    {'token': 'they', 'doc_count': 1, 'token_count': 1, 'token_rank': 8.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.631578947368421, 'doc_pct': 0.3333333333333333},
    #    {'token': 'got', 'doc_count': 1, 'token_count': 1, 'token_rank': 9.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.6842105263157894, 'doc_pct': 0.3333333333333333},
    #    {'token': 'tired', 'doc_count': 1, 'token_count': 1, 'token_rank': 10.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.7368421052631577, 'doc_pct': 0.3333333333333333},
    #    {'token': 'and', 'doc_count': 1, 'token_count': 1, 'token_rank': 11.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.7894736842105261, 'doc_pct': 0.3333333333333333},
    #    {'token': 'went', 'doc_count': 1, 'token_count': 1, 'token_rank': 12.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.8421052631578945, 'doc_pct': 0.3333333333333333},
    #    {'token': 'to', 'doc_count': 1, 'token_count': 1, 'token_rank': 13.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.8947368421052628, 'doc_pct': 0.3333333333333333},
    #    {'token': 'sleep', 'doc_count': 1, 'token_count': 1, 'token_rank': 14.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.9473684210526312, 'doc_pct': 0.3333333333333333},
    #    {'token': 'zzz', 'doc_count': 1, 'token_count': 1, 'token_rank': 15.0, 'token_pct': 0.05263157894736842, 'token_pct_rt': 0.9999999999999996, 'doc_pct': 0.3333333333333333}
    #]
