import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
import pickle as pickle
from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# finetuning
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    AutoConfig,
)

# Evaluation
import datasets
from datasets import load_metric, load_dataset
from sklearn.metrics import f1_score
from huggingface_hub import notebook_login

DATASET = "./../data/processed/data.csv"
TEST_DATASET = "./../data/datasets/test.csv"


def load_data():
    df = pd.read_csv(DATASET)
    df.comment = df.comment.astype(str)
    test_df = pd.read_csv(TEST_DATASET)
    test_df.comment = test_df.comment.astype(str)
    return df, test_df


def train_distilbert_model(train_df):
    train_df = pd.DataFrame(train_df).dropna()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    train, eval = train_test_split(train_df, train_size=0.8, stratify=train_df["label"])

    pd.DataFrame(train).to_csv("../data/datasets/distilbert/train.csv", index=False)
    pd.DataFrame(eval).to_csv("../data/datasets/distilbert/eval.csv", index=False)

    datasets = load_dataset(
        "csv",
        data_files={
            "train": "../data/datasets/distilbert/train.csv",
            "eval": "../data/datasets/distilbert/eval.csv",
        },
    )

    def tokenize_data(data):
        return tokenizer(
            [str(comment) for comment in data["comment"] if comment is not None],
            padding="max_length",
        )

    # def transform_label(data):
    #     label = data["label"]
    #     num = [0.0, 0.0, 1.0]
    #     if label == "negative":  # Negative
    #         num = [0.0, 0.0, 1.0]
    #     elif label == "neutral":  # Neutral
    #         num = [0.0, 1.0, 0.0]
    #     else:  # Positive
    #         num = [1.0, 0.0, 0.0]
    #     return {"labels": num}

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"f1-score": f1}

    datasets = datasets.map(tokenize_data, batched=True, batch_size=256)
    # remove_columns = ["comment", "label"]
    # datasets = datasets.map(transform_label, remove_columns=remove_columns)

    train_dataset = datasets["train"].shuffle(seed=0)
    eval_dataset = datasets["eval"].shuffle(seed=0)

    trainargs = TrainingArguments(
        "job_reviews_sentiment_analysis_distilbert",
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    model_name = "distilbert-base-cased"
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    trainer = Trainer(
        model=model,
        args=trainargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print("DistilBERT model")
    print("Evaluation results:", results)

    return trainer, tokenizer


def save_distilbert_model(model, tokenizer):
    model.save("../models/distilbert_sentiment_analysis.h5")
    with open("tokenizer-distilbert.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_custom_model(train_df):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["comment"])
    sequences = tokenizer.texts_to_sequences(train_df["comment"])
    padded_sequences = pad_sequences(sequences, maxlen=100, truncating="post")
    sentiment_labels = pd.get_dummies(train_df["label"]).values

    x_train, x_eval, y_train, y_eval = train_test_split(
        padded_sequences, sentiment_labels, test_size=0.2
    )

    model = Sequential()
    model.add(Embedding(5000, 100, input_length=100))
    model.add(Conv1D(64, 5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        x_train, y_train, epochs=3, batch_size=256, validation_data=(x_eval, y_eval)
    )

    y_pred = np.argmax(model.predict(x_eval), axis=-1)
    print("Accuracy:", accuracy_score(np.argmax(y_eval, axis=-1), y_pred))

    return model, tokenizer


def save_custom_model(model, tokenizer):
    model.save("../models/sentiment_analysis_model.h5")
    with open("../models/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_distilbert_model(test_df):
    model = load_model("../models/distilbert_sentiment_analysis.h5")
    with open("../models/tokenizer-distilbert.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def tokenize_data(data):
        return tokenizer(data["comment"], padding="max_length")

    test_df = pd.DataFrame(test_df).map(tokenize_data, batched=True)
    predicted_rating = model.predict(test_df["comment"])[0]


def test_custom_model(test_df):
    model = load_model("../models/sentiment_analysis_model.h5")
    with open("../models/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences(test_df["comment"])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)
    array = np.argmax(predicted_rating, axis=1)

    def categories_to_string(cat):
        if cat == 0:
            return "Positive"
        elif cat == 1:
            return "Neutral"
        else:
            return "Negative"

    vector_func = np.vectorize(categories_to_string)
    test_df["custom_model_sentiment"] = vector_func(array)

    return test_df


def perform_textblob(test_df):
    tb = Blobber(analyzer=NaiveBayesAnalyzer())

    def classification_score(classification):
        if classification == "pos":
            return "Positive"
        elif classification == "neg":
            return "Negative"
        else:
            return "Neutral"

    def sentiment_calc(text):
        try:

            return classification_score(tb(text).sentiment.classification)
        except:
            return None

    test_df["textblob_sentiment"] = test_df["comment"].apply(sentiment_calc)

    return test_df


def perform_vader(test_df):
    def polarity_score(compound):
        if compound > 0.05:
            return "Positive"
        elif compound < -0.5:
            return "Negative"
        elif compound >= -0.05 and compound < 0.05:
            return "Neutral"

    def sentiment_calc(text):
        analyzer = SentimentIntensityAnalyzer()
        try:
            return polarity_score(analyzer.polarity_scores(text)["compound"])
        except:
            return None

    test_df["vader_sentiment"] = test_df["comment"].apply(sentiment_calc)

    return test_df


def write(test_df):
    test_df.to_csv("./../data/datasets/result.csv", index=False)


def main():
    print("Loading data...")
    train_df, test_df = load_data()

    # print("Training custom model...")
    # model, tokenizer = train_custom_model(train_df)
    # save_custom_model(model, tokenizer)

    # print("Testing distilbert model...")
    # model, tokenizer = train_distilbert_model(train_df)
    # save_distilbert_model(model, tokenizer)

    print("Performing custom model...")
    test_df = pd.DataFrame(test_custom_model(test_df))
    # print("Performing distilbert model ...")
    # test_df = pd.DataFrame(test_distilbert_model(test_df))
    print("Performing TextBlob pre-trained model...")
    test_df = pd.DataFrame(perform_textblob(test_df))
    print("Performing VADER pre-trained model...")
    test_df = pd.DataFrame(perform_vader(test_df))
    print("Writing .csv file...")
    write(test_df)
    print("Finsihed.")


if __name__ == "__main__":
    main()
