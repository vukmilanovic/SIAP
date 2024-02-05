import pandas as pd
import numpy as np
import re

GLASSDOOR_DATASET = "./../data/raw/glassdoor_reviews.csv"
EMPLOYEES_TEST_DATASET = "./../data/raw/test.csv"
EMPLOYEES_TRAIN_DATASET = "./../data/raw/train.csv"
PATTERN = r"[^a-zA-Z\s]"


def load():
    return (
        pd.read_csv(GLASSDOOR_DATASET, index_col=0),
        pd.read_csv(EMPLOYEES_TEST_DATASET, index_col=0),
        pd.read_csv(EMPLOYEES_TRAIN_DATASET, index_col=0),
    )


def set_comment(row):
    if row["overall_rating"] > 3:
        return row["pros"]
    elif row["overall_rating"] < 3:
        return row["cons"]
    else:
        return row["summary"]


def set_label(row):
    if row["overall_rating"] > 3:
        return 2
    elif row["overall_rating"] < 3:
        return 0
    else:
        return 1


def process_glassdoor_df(glassdoor_ds):
    # Selecting columns of interest
    glassdoor_ds = glassdoor_ds.filter(
        ["firm", "headline", "overall_rating", "pros", "cons"], axis=1
    )

    # Applying regex to remove all characters except lowercase, uppercase and whitespaces
    columns = ["headline", "pros", "cons"]
    glassdoor_ds[columns] = glassdoor_ds[columns].apply(
        lambda x: x.str.replace(PATTERN, "", regex=True)
    )

    # Dropping NaN values
    glassdoor_ds["headline"].replace("", np.nan, inplace=True)
    glassdoor_ds["pros"].replace("", np.nan, inplace=True)
    glassdoor_ds["cons"].replace("", np.nan, inplace=True)

    glassdoor_ds.dropna(subset=columns, inplace=True)

    # Dropping rows where review comment has 1 char or less
    glassdoor_ds = glassdoor_ds[glassdoor_ds["headline"].str.len() > 1]
    glassdoor_ds = glassdoor_ds[glassdoor_ds["pros"].str.len() > 1]
    glassdoor_ds = glassdoor_ds[glassdoor_ds["cons"].str.len() > 1]

    # Renaming dataframe column
    glassdoor_ds.rename(columns={"headline": "summary"}, inplace=True)

    # Creating the 'comment' column
    glassdoor_ds["comment"] = glassdoor_ds.apply(set_comment, axis=1)
    glassdoor_ds["label"] = glassdoor_ds.apply(set_label, axis=1)

    # Dropping unnecessary columns
    glassdoor_ds.drop(
        columns=["pros", "cons", "summary", "overall_rating"], inplace=True
    )

    return pd.DataFrame(glassdoor_ds)


def process_employee_reviews_df(e_test_df, e_train_df):
    # Selecting columns of interest
    e_test_df = pd.DataFrame(
        e_test_df[
            [
                "summary",
                "positives",
                "negatives",
                "score_1",
                "score_2",
                "score_3",
                "score_4",
                "score_5",
            ]
        ]
    )
    e_train_df = pd.DataFrame(
        e_train_df[["summary", "positives", "negatives", "overall"]].rename(
            columns={
                "overall": "overall_rating",
                "positives": "pros",
                "negatives": "cons",
            }
        )
    )
    e_train_df.dropna(subset=["overall_rating"], inplace=True)

    # Calculating overall scores in test dataset
    score_columns = ["score_1", "score_2", "score_3", "score_4", "score_5"]
    e_test_df.dropna(subset=score_columns, inplace=True)
    e_test_df["overall_rating"] = e_test_df[score_columns].mean(axis=1).round()

    # Dropping unnecessary columns
    e_test_df.drop(columns=score_columns, inplace=True)
    e_test_df.rename(columns={"positives": "pros", "negatives": "cons"}, inplace=True)

    # Concatenating two dataframes
    result_df = pd.concat([e_test_df, e_train_df], ignore_index=True)

    # Applying regex to remove all characters except lowercase, uppercase and whitespaces
    columns = ["summary", "pros", "cons"]
    result_df[columns] = result_df[columns].apply(
        lambda x: x.str.replace(PATTERN, "", regex=True)
    )

    # Dropping NaN values
    result_df["summary"].replace("", np.nan, inplace=True)
    result_df["pros"].replace("", np.nan, inplace=True)
    result_df["cons"].replace("", np.nan, inplace=True)

    result_df.dropna(subset=["summary"], inplace=True)

    # Dropping rows where review comment has 1 char or less
    result_df = result_df[result_df["summary"].str.len() > 1]
    result_df = result_df[result_df["pros"].str.len() > 1]
    result_df = result_df[result_df["cons"].str.len() > 1]

    # Creating the 'comment' column
    result_df["comment"] = result_df.apply(set_comment, axis=1)
    result_df["label"] = result_df.apply(set_label, axis=1)
    result_df.drop(columns=["summary", "pros", "cons", "overall_rating"], inplace=True)

    return pd.DataFrame(result_df)


def setup_datasets(gd_df, er_df):
    # Concatenating two processed datasets
    result_df = pd.concat([gd_df, er_df], ignore_index=True)

    # Removing some additional characters from comment column
    result_df["comment"] = result_df["comment"].str.replace("\n", "")
    result_df["comment"] = result_df["comment"].str.replace(" +", " ")
    result_df["comment"] = result_df["comment"].str.replace('"', "")
    result_df["comment"] = result_df["comment"].str.strip()

    # Sampling result_df and creating test dataset
    test_df = result_df.sample(frac=0.15, random_state=42)
    test_df.reset_index(drop=True, inplace=True)
    other_df = result_df.drop(test_df.index)
    other_df.reset_index(drop=True, inplace=True)

    # Sampling other_df and creating training and validation datasets
    validation_df = other_df.sample(frac=0.2, random_state=42)
    validation_df.reset_index(drop=True, inplace=True)
    train_df = other_df.drop(validation_df.index)
    train_df.reset_index(drop=True, inplace=True)

    # Dropping labal columns in test and validation datasets
    test_df.drop(columns=["label"], inplace=True)
    validation_df.drop(columns=["label"], inplace=True)

    return train_df, test_df, validation_df


def write(train_df, test_df, validation_df):
    train_df.to_csv("./../data/train-dataset/train.csv", index=False)
    test_df.to_csv("./../data/test-dataset/test.csv", index=False)
    validation_df.to_csv("./../data/validation-dataset/validation.csv", index=False)


def main():
    glassdoor_df, e_test_df, e_train_df = load()
    gd_df = process_glassdoor_df(glassdoor_df)
    er_df = process_employee_reviews_df(e_test_df, e_train_df)
    train_df, test_df, validation_df = setup_datasets(gd_df, er_df)
    write(train_df, test_df, validation_df)


if __name__ == "__main__":
    main()
