

import os

import pandas as pd

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEWS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "amazon_reviews.csv")

if __name__ == "__main__":
    df = pd.read_csv(REVIEWS_CSV_FILEPATH)

    print(df.head())

    breakpoint()
