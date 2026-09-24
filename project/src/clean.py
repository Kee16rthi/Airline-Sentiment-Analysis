"""
clean.py
Cleaning utilities for the Twitter Airline Sentiment dataset.
"""

import re
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and keep only the columns we need."""
    df = pd.read_csv(path)
    df = df[["airline_sentiment", "airline", "text", "negativereason"]].copy()
    df = df.dropna(subset=["text", "airline_sentiment"])
    df = df.drop_duplicates(subset=["text"])
    return df


def clean_text(text: str) -> str:
    """
    Clean a single tweet:
    - lowercase
    - remove URLs
    - remove @mentions (they're noise for sentiment, e.g. @VirginAmerica)
    - remove extra whitespace
    - keep hashtag words but drop the '#' symbol (hashtag text often carries sentiment)
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                     # @mentions
    text = re.sub(r"#", "", text)                          # keep hashtag word, drop symbol
    text = re.sub(r"[^a-z0-9\s']", " ", text)              # strip punctuation/emoji except apostrophes
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to the whole dataframe and drop rows that become empty."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


if __name__ == "__main__":
    df = load_data("data/Tweets.csv")
    df = clean_dataframe(df)
    print(f"Rows after cleaning: {len(df)}")
    print(df[["text", "clean_text", "airline_sentiment"]].head(5))
    df.to_csv("data/Tweets_clean.csv", index=False)
    print("Saved cleaned data to data/Tweets_clean.csv")
