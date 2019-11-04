
# FYI CATEGORIES = {1: "Scotch", 2: "Bourbon/Tennessee", 3: "Craft Whiskey", 4: "Canadian"}

import os

import pandas as pd

WHISKEY_DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "whiskey")
SUBMISSION_CSV_FILEPATH = os.path.join(WHISKEY_DATA_DIRPATH, "my_submission.csv")

def generate_submissions(model, csv_filepath=SUBMISSION_CSV_FILEPATH):
    testing_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "test.csv"))

    submission_df = testing_df

    # use the model to predict categories for all texts in the testing set

    breakpoint()

    # then write those to SUBMISSION_CSV_FILEPATH
    # TODO: create an example submission CSV file resembling:
    # id,category
    # 955,1
    # 3532,3
    # 1390,2
    # 1024,4
    # and return the dataframe for good measure

    return submission_df

if __name__ == "__main__":

    training_df = pd.read_csv(os.path.join(WHISKEY_DATA_DIRPATH, "train.csv"))
    print(training_df.head())
    print(training_df["category"].value_counts())

    for i, row in training_df.iterrows():
        print(row["id"], row["category"], row["description"][0:90])

    #breakpoint()
