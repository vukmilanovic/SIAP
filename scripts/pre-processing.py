import pandas as pd
import numpy as np

GLASSDOOR_DATASET = (
    r"C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/glassdoor_reviews.csv"
)
EMPLOYEES_TEST_DATASET = r"C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/test.csv"
EMPLOYEES_TRAIN_DATASET = (
    r"C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/train.csv"
)


def load():
    return (
        pd.read_csv(GLASSDOOR_DATASET, index_col=0),
        pd.read_csv(EMPLOYEES_TEST_DATASET, index_col=0),
        pd.read_csv(EMPLOYEES_TRAIN_DATASET, index_col=0),
    )


def process_glassdoor_df(glassdoor_ds):
    # Selecting columns of interest
    glassdoor_ds = glassdoor_ds.filter(["firm", "headline"], axis=1)

    # Applying regex to remove all characters except lowercase, uppercase and whitespaces
    glassdoor_ds = pd.DataFrame(
        glassdoor_ds["headline"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    )

    # Dropping NaN values
    glassdoor_ds["headline"].replace("", np.nan, inplace=True)
    glassdoor_ds.dropna(subset=["headline"], inplace=True)

    # Dropping rows where review comment has 1 char or less
    glassdoor_ds = glassdoor_ds[glassdoor_ds["headline"].str.len() > 1]

    # Renaming dataframe column
    glassdoor_ds.rename(columns={"headline": "summary"}, inplace=True)

    return pd.DataFrame(glassdoor_ds["summary"])


def process_employee_reviews_df(e_test_df, e_train_df):
    # Selecting columns of interest
    e_test_df = e_test_df[["summary"]]
    e_train_df = e_train_df[["summary"]]

    # Concatenating two dataframes
    result_df = pd.concat([e_test_df, e_train_df], ignore_index=True)

    # Applying regex to remove all characters except lowercase, uppercase and whitespaces
    result_df = pd.DataFrame(
        result_df["summary"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    )

    # Dropping NaN values
    result_df["summary"].replace("", np.nan, inplace=True)
    result_df.dropna(subset=["summary"], inplace=True)

    # Dropping rows where review comment has 1 char or less
    result_df = result_df[result_df["summary"].str.len() > 1]

    return pd.DataFrame(result_df)


def setup_datasets(gd_df, er_df):
    # Concatenating two processed datasets
    result_df = pd.concat([gd_df, er_df], ignore_index=True)

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

    return train_df, test_df, validation_df


def write(train_df, test_df, validation_df):
    train_df.to_csv("./../data/train_dataset/train.csv", index=False)
    test_df.to_csv("./../data/test_dataset/test.csv", index=False)
    validation_df.to_csv("./../data/validation_dataset/validation.csv", index=False)


def main():
    glassdoor_df, e_test_df, e_train_df = load()
    gd_df = process_glassdoor_df(glassdoor_df)
    er_df = process_employee_reviews_df(e_test_df, e_train_df)
    train_df, test_df, validation_df = setup_datasets(gd_df, er_df)
    write(train_df, test_df, validation_df)


if __name__ == "__main__":
    main()
