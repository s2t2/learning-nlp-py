
# Learning NLP

## Setup

Fork and clone this repo onto your local machine, then navigate there from the command-line.

```sh
conda create -n learning-nlp-env python=3.7 # (first time only)
conda activate learning-nlp-env
```

```sh
pip install -r requirements.txt
```

Download the "amazon_reviews.csv" file and move it into the "data" directory of this repository.

## Usage

```sh
python -m app.parser
```

## Testing

```sh
pip install pytest # (first time only)
```

```sh
pytest
```
