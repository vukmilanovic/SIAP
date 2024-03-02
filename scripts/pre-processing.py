import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections
import re
import seaborn

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
        return [1.0, 0.0, 0.0]
    elif row["overall_rating"] < 3:
        return [0.0, 0.0, 1.0]
    else:
        return [0.0, 1.0, 0.0]


def convert_labels(row):
    if row == [1.0, 0.0, 0.0]:
        return "Positive"
    elif row == [0.0, 0.0, 1.0]:
        return "Negative"
    else:
        return "Neutral"


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
    result_df["la bel"] = result_df.apply(set_label, axis=1)
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

    test_df.drop(columns=["label"], inplace=True)

    comments = " ".join(other_df["comment"].astype(str))
    words = re.findall(r"\b\w+\b", comments)
    words = [word.lower() for word in words]

    word_counts = collections.Counter(words)

    combined_words = []
    for word in words:
        if word.endswith("s") and len(word) > 1:
            singular_word = word[:-1]
            if singular_word in word_counts:
                combined_words.append(singular_word)
            else:
                combined_words.append(word)
        elif len(word) > 1:
            combined_words.append(word)

    word_counts = collections.Counter(combined_words)
    top_words = dict(word_counts.most_common(20))

    plt.figure(figsize=(10, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top 20 Most Common Words")
    plt.xticks(rotation=45)
    plt.show()

    # Separate positive, negative and neutral comments based on the 'label' column
    positive_comments = other_df[
        other_df["label"].apply(lambda x: x == [1.0, 0.0, 0.0])
    ].sample(frac=0.5, random_state=23)
    neutral_comments = other_df[
        other_df["label"].apply(lambda x: x == [0.0, 1.0, 0.0])
    ].sample(frac=0.5, random_state=23)
    negative_comments = other_df[
        other_df["label"].apply(lambda x: x == [0.0, 0.0, 1.0])
    ].sample(frac=0.5, random_state=23)

    # Sampling to create word clouds
    sample_positive_text = " ".join(positive_comments["comment"])
    sample_neutral_text = " ".join(neutral_comments["comment"])
    sample_negative_text = " ".join(negative_comments["comment"])

    # Generate word cloud images for every type of sentiment
    wordcloud_positive = WordCloud(
        width=1000, height=600, max_words=300, background_color="white"
    ).generate(sample_positive_text)
    wordcloud_neutral = WordCloud(
        width=1000, height=600, max_words=300, background_color="white"
    ).generate(sample_neutral_text)
    wordcloud_negative = WordCloud(
        width=1000, height=600, max_words=300, background_color="white"
    ).generate(sample_negative_text)

    # Display the generated image using mathplotlib
    plt.figure(figsize=(18, 6))

    # Positive word cloud
    plt.subplot(1, 3, 1)
    plt.imshow(wordcloud_positive, interpolation="bilinear")
    plt.title("Positive Comments Word Cloud")
    plt.axis("off")

    # Neutral word cloud
    plt.subplot(1, 3, 2)
    plt.imshow(wordcloud_neutral, interpolation="bilinear")
    plt.title("Neutral Comments Word Cloud")
    plt.axis("off")

    # Negative word cloud
    plt.subplot(1, 3, 3)
    plt.imshow(wordcloud_negative, interpolation="bilinear")
    plt.title("Negative Comments Word Cloud")
    plt.axis("off")

    plt.show()

    # Calculate the frequency of each sentiment label
    plt.figure(figsize=(10, 6))
    converted_labels = other_df["label"].apply(convert_labels).astype("category")
    seaborn.barplot(
        x=converted_labels.value_counts().index,
        y=converted_labels.value_counts().values,
    )
    plt.xlabel("Sentiment Label")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sentiment Labels")
    plt.show()

    # Calculate the length of each text in the "safe_text" column
    text_lengths = other_df["comment"].apply(len)
    plt.figure(figsize=(8, 6))
    seaborn.histplot(text_lengths, kde=True)
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Text Lengths")
    plt.show()

    return other_df, test_df


def write(df, validation_df):
    df.to_csv("./../data/processed/data.csv", index=False)
    validation_df.to_csv("./../data/datasets/test.csv", index=False)


def main():
    glassdoor_df, e_test_df, e_train_df = load()
    gd_df = process_glassdoor_df(glassdoor_df)
    er_df = process_employee_reviews_df(e_test_df, e_train_df)
    df, validation_df = setup_datasets(gd_df, er_df)
    write(df, validation_df)


if __name__ == "__main__":
    main()
