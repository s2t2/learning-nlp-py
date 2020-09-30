# What is NLP?

It is a platform for building program to work with human language data. Vision behind the introduction of Natural language libraries to interact with human conversation and respond back in his/her understandable form. There are many natural language processing libraries available such as SpaCy, Gensim, Pattern, TextBlob and PolyGlot and Natural Language Toolkit. This article is all about guiding readers with NLTK, its modules and sub modules in an effective and visualized way. Text mining python is one of the area where research is very effective and such is widely used in sentimental analysis. Please see the different types of libraries which comes under NLP.

# Working Pipeline of NLP: 
![image](https://drive.google.com/uc?export=view&id=1wvU0rsuMDIc1qsM5yZr3UqL4E0S_S28a)
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
  + Mod 3: Download the data from this [Kaggle Competition](https://www.kaggle.com/c/whiskey-201911/data), and move it into the "data/whiskey" directory of this repository. (FYI: ALREADY INCLUDED IN THIS REPO)
  + Mod 4: Text snippets from novels already included in the "data/novels" directory of this repository. (FYI: ALREADY INCLUDED IN THIS REPO)

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
> nltk.download("movie_reviews")
```
## Usage

Run some example code:

```sh
# MOD 1:
python -m app.tokenizer

# MOD 2:
python -m app.vectorizer
python -m app.word_distances

# MOD 3:
python -m app.grid_searcher
python -m app.amzn_reviews_classifier
python -m app.imdb_reviews_classifier
python -m app.whiskey_reviews_classifier

# MOD 4:
python -m app.novels
```

Start working from scratch in your own clean space:

```sh
python -m app.playground # MOD 1
python -m app.playground2 # MOD 2
python -m app.playground3 # MOD 3
python -m app.playground4 # MOD 4
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
