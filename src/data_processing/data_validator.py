"""
src/data_processing/data_validator.py
Data quality checks for the BBC News dataset and processed DataFrames.
"""

import logging
import pandas as pd
import numpy as np
from config.settings import CATEGORIES, TEXT_COLUMN, LABEL_COLUMN

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = [TEXT_COLUMN, LABEL_COLUMN]
REQUIRED_PROCESSED_COLUMNS = [
    "cleaned_text", "category", "tokens", "entities",
    "sentiment_label", "sentiment_score",
]
MIN_TEXT_LENGTH = 50   # characters


class DataValidationError(Exception):
    """Raised when a critical data quality check fails."""
    pass


def validate_raw_dataframe(df: pd.DataFrame) -> dict:
    """
    Run quality checks on the raw BBC News DataFrame.

    Checks:
        - Required columns present
        - No fully null rows
        - Valid category labels
        - Minimum text length
        - Duplicate detection

    Args:
        df: Raw loaded DataFrame.

    Returns:
        Dict with keys: valid (bool), warnings (list), stats (dict).

    Raises:
        DataValidationError: If a critical check fails.
    """
    report = {"valid": True, "warnings": [], "stats": {}}

    # Check required columns
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # Shape
    report["stats"]["n_rows"] = len(df)
    report["stats"]["n_cols"] = len(df.columns)

    # Null check
    null_counts = df[REQUIRED_RAW_COLUMNS].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            report["warnings"].append(f"Column '{col}' has {count} null values.")

    # Category validity
    invalid_cats = set(df[LABEL_COLUMN].dropna().unique()) - set(CATEGORIES)
    if invalid_cats:
        report["warnings"].append(f"Unexpected categories: {invalid_cats}")
    report["stats"]["category_distribution"] = df[LABEL_COLUMN].value_counts().to_dict()

    # Minimum text length
    short_texts = (df[TEXT_COLUMN].str.len() < MIN_TEXT_LENGTH).sum()
    if short_texts > 0:
        report["warnings"].append(f"{short_texts} articles below {MIN_TEXT_LENGTH} chars.")

    # Duplicate texts
    dupes = df[TEXT_COLUMN].duplicated().sum()
    if dupes > 0:
        report["warnings"].append(f"{dupes} duplicate articles detected.")

    report["stats"]["duplicate_count"] = int(dupes)
    report["stats"]["avg_text_length"] = float(df[TEXT_COLUMN].str.len().mean())

    if report["warnings"]:
        logger.warning(f"Validation warnings: {report['warnings']}")
    else:
        logger.info("Raw DataFrame passed all quality checks.")

    return report


def validate_processed_dataframe(df: pd.DataFrame) -> dict:
    """
    Run quality checks on a preprocessed DataFrame (df_final).

    Args:
        df: Processed DataFrame expected to have NLP feature columns.

    Returns:
        Validation report dict.
    """
    report = {"valid": True, "warnings": [], "stats": {}}

    missing = [c for c in REQUIRED_PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        report["warnings"].append(f"Missing processed columns: {missing}")
        report["valid"] = False

    # Token length distribution
    if "tokens" in df.columns:
        token_lens = df["tokens"].apply(len)
        report["stats"]["avg_token_count"] = float(token_lens.mean())
        report["stats"]["min_token_count"] = int(token_lens.min())
        empty_docs = (token_lens == 0).sum()
        if empty_docs > 0:
            report["warnings"].append(f"{empty_docs} documents have zero tokens after preprocessing.")

    # Sentiment coverage
    if "sentiment_label" in df.columns:
        dist = df["sentiment_label"].value_counts().to_dict()
        report["stats"]["sentiment_distribution"] = dist

    report["stats"]["shape"] = df.shape

    logger.info(f"Processed DataFrame validation: {report}")
    return report


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Print a human-readable dataset summary to stdout.

    Args:
        df: Raw or processed DataFrame.
    """
    print("=" * 55)
    print("  NEWSBOT DATASET SUMMARY")
    print("=" * 55)
    print(f"  Shape           : {df.shape}")
    print(f"  Columns         : {list(df.columns)}")
    if LABEL_COLUMN in df.columns:
        print(f"\n  Category distribution:")
        for cat, cnt in df[LABEL_COLUMN].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {cat:<15} {cnt:5d}  ({pct:.1f}%)")
    if TEXT_COLUMN in df.columns:
        lens = df[TEXT_COLUMN].str.len()
        print(f"\n  Text length:")
        print(f"    Mean  : {lens.mean():.0f} chars")
        print(f"    Min   : {lens.min()} chars")
        print(f"    Max   : {lens.max()} chars")
    print("=" * 55)
