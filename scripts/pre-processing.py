import pandas as pd

GLASSDOOR_DATASET = r'C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/glassdoor_reviews.csv'
EMPLOYEES_TEST_DATASET = r'C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/test.csv'
EMPLOYEES_TRAIN_DATASET = r'C://Users/One/Desktop/Master/SIAP/Projekat/data/raw/train.csv'


def load():
    return pd.read_csv(GLASSDOOR_DATASET, index_col=0), \
        pd.read_csv(EMPLOYEES_TEST_DATASET, index_col=0), \
        pd.read_csv(EMPLOYEES_TRAIN_DATASET, index_col=0)


def processing():
    pass


def write():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
