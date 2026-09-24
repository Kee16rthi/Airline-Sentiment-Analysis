"""
eda.py
Exploratory data analysis for the Twitter Airline Sentiment dataset.
Generates and saves 3 plots to the notebooks/ folder for use in the README.
"""

import pandas as pd
import matplotlib.pyplot as plt
from clean import load_data, clean_dataframe

plt.rcParams["figure.dpi"] = 120


def plot_class_distribution(df: pd.DataFrame, out_path: str):
    counts = df["airline_sentiment"].value_counts()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar", color=["#c0392b", "#7f8c8d", "#27ae60"])
    plt.title("Sentiment Class Distribution")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_airline_breakdown(df: pd.DataFrame, out_path: str):
    ct = pd.crosstab(df["airline"], df["airline_sentiment"])
    ct = ct[["negative", "neutral", "positive"]]
    ct.plot(kind="bar", stacked=True, color=["#c0392b", "#7f8c8d", "#27ae60"], figsize=(7, 4))
    plt.title("Sentiment by Airline")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_text_length(df: pd.DataFrame, out_path: str):
    df = df.copy()
    df["word_count"] = df["clean_text"].str.split().apply(len)
    plt.figure(figsize=(6, 4))
    df.boxplot(column="word_count", by="airline_sentiment")
    plt.title("Tweet Length by Sentiment")
    plt.suptitle("")
    plt.ylabel("Word Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def top_negative_reasons(df: pd.DataFrame, n: int = 10):
    reasons = df[df["airline_sentiment"] == "negative"]["negativereason"].value_counts().head(n)
    print("\nTop reasons for negative sentiment:")
    print(reasons)
    return reasons


if __name__ == "__main__":
    raw = load_data("data/Tweets.csv")
    df = clean_dataframe(raw)

    print(f"Total tweets after cleaning: {len(df)}")
    print("\nClass balance:")
    print(df["airline_sentiment"].value_counts(normalize=True).round(3))

    plot_class_distribution(df, "notebooks/class_distribution.png")
    plot_airline_breakdown(df, "notebooks/airline_breakdown.png")
    plot_text_length(df, "notebooks/text_length.png")
    top_negative_reasons(df)
