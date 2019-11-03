
# Learning NLP

![](/img/tokenization-v5.png)

![](/img/word_dependencies.png)

## Setup

Fork this repo and clone your forked copy onto your local machine, then navigate there from the command-line:

```sh
cd learning-nlp-py/
```

Create and/or activate a Python 3.7 virtual environment:

```sh
conda create -n learning-nlp-env python=3.7 # (first time only)
conda activate learning-nlp-env
```

Install package dependencies:

```sh
pip install -r requirements.txt # (first time only)
```

Download the data:

  + Mod 1: Download the "amazon_reviews.csv" file and move it into the "data" directory of this repository.
  + Mod 2: Download the "bbc_docs" directory of text files, and move it into the "data" directory of this repository.

Download the spacy language models:

```sh
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

Download NLTK data, like stopwords:

```sh
python

> import nltk
> nltk.download()
> nltk.download("stopwords")
```
## Usage

Run some example code:

```sh
python -m app.tokenizer
python -m app.vectorizer
python -m app.word_distances
python -m app.amzn_reviews_classifier -W
```

Start working from scratch in your own clean space:

```sh
python -m app.playground
python -m app.playground2
```

## Testing

```sh
pip install pytest # (first time only)
```

```sh
pytest
# pytest --disable-pytest-warnings -s
# pytest test/parser_test.py --disable-pytest-warnings -s
# pytest test/parser_test.py --disable-pytest-warnings -s -k 'test_tokenize'
```
