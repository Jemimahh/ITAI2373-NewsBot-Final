"""
src/analysis/sentiment_analyzer.py
VADER-based sentiment analysis with enhanced category-level aggregation.
Extended from ITAI 2373 midterm Module 6.
"""

import logging
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config.settings import VADER_POSITIVE_THRESHOLD, VADER_NEGATIVE_THRESHOLD

logger = logging.getLogger(__name__)

_analyzer: SentimentIntensityAnalyzer = None


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return singleton VADER analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_sentiment(text: str) -> dict:
    """
    Run VADER sentiment analysis on a text string.

    Args:
        text: Input article text.

    Returns:
        Dict with keys: compound, pos, neu, neg, label.
    """
    scores = get_analyzer().polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= VADER_POSITIVE_THRESHOLD:
        label = "Positive"
    elif compound <= VADER_NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "compound":        compound,
        "positive":        scores["pos"],
        "neutral":         scores["neu"],
        "negative":        scores["neg"],
        "sentiment_label": label,
    }


def analyze_dataframe(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Apply VADER sentiment analysis across a DataFrame.

    Adds columns: sentiment_compound, sentiment_label.

    Args:
        df: DataFrame with text column.
        text_col: Name of column to analyze.

    Returns:
        DataFrame with new sentiment columns.
    """
    df = df.copy()
    results = df[text_col].apply(analyze_sentiment)
    df["sentiment_compound"] = results.apply(lambda r: r["compound"])
    df["sentiment_label"]    = results.apply(lambda r: r["sentiment_label"])
    logger.info("Sentiment analysis complete.")
    return df


def sentiment_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment statistics per news category.

    Returns a DataFrame with mean compound score and label distribution.
    """
    agg = df.groupby("category")["sentiment_compound"].agg(["mean", "std", "count"])
    label_dist = df.groupby("category")["sentiment_label"].value_counts(normalize=True).unstack(fill_value=0)
    return agg.join(label_dist)


def track_sentiment_evolution(
    df: pd.DataFrame,
    time_col: str = "category",
) -> pd.DataFrame:
    """
    Track sentiment changes across a time or category axis.

    If a real date column is available, pass its name as time_col.
    Falls back to category as a proxy ordering axis.

    Args:
        df: DataFrame with sentiment_compound column.
        time_col: Column to group by (date or category).

    Returns:
        Aggregated DataFrame sorted by time_col.
    """
    return (
        df.groupby(time_col)["sentiment_compound"]
        .agg(["mean", "std", "min", "max"])
        .rename(columns={"mean": "avg_sentiment", "std": "sentiment_volatility"})
    )
