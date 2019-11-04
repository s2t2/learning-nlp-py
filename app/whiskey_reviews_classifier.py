
import os

import pandas as pd

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "whiskey")

if __name__ == "__main__":

    training_df = pd.read_csv(os.path.join(DATA_DIRPATH, "train.csv"))
    testing_df = pd.read_csv(os.path.join(DATA_DIRPATH, "test.csv"))

    print(training_df.head())

    #breakpoint()

    # TODO: create an example submission CSV file resembling:
    # id,category
    # 955,1
    # 3532,3
    # 1390,2
    # 1024,4
