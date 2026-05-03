"""
src/multilingual/cross_lingual_analyzer.py
Cross-language sentiment and topic comparison.
Module C — Multilingual Intelligence.
"""

import logging
import pandas as pd
from src.multilingual.translator import translate_text
from src.analysis.sentiment_analyzer import analyze_sentiment
from src.data_processing.text_preprocessor import clean_text

logger = logging.getLogger(__name__)

COMPARISON_LANGUAGES = ["fr", "de", "es", "it"]


def cross_lingual_sentiment(
    article_text: str,
    languages: list[str] = None,
) -> dict:
    """
    Analyze sentiment of an article's translation in multiple languages.

    Translates the article to each target language, then back-translates
    to English and analyzes VADER sentiment on the round-trip text.
    This reveals how translation may shift emotional tone.

    Args:
        article_text: Original English article text.
        languages: List of target language codes. Defaults to COMPARISON_LANGUAGES.

    Returns:
        Dict mapping language_code → sentiment result dict.
    """
    targets = languages or COMPARISON_LANGUAGES
    results = {"en": analyze_sentiment(article_text)}

    for lang in targets:
        try:
            # Translate to target language
            fwd = translate_text(article_text[:1500], target_lang=lang)
            # Back-translate to English for VADER analysis
            back = translate_text(fwd["translated_text"], target_lang="en", source_lang=lang)
            sentiment = analyze_sentiment(back["translated_text"])
            sentiment["translated_preview"] = fwd["translated_text"][:200]
            results[lang] = sentiment
        except Exception as e:
            logger.warning(f"Cross-lingual sentiment failed for {lang}: {e}")
            results[lang] = {"error": str(e)}

    return results


def compare_coverage(
    df: pd.DataFrame,
    topic_keyword: str,
    source_col: str = "cleaned_text",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Compare how a topic is covered across BBC news categories.

    Filters articles mentioning a keyword and compares sentiment
    and volume across categories (proxy for editorial perspective).

    Args:
        df: DataFrame with article text and category columns.
        topic_keyword: Keyword to filter articles.
        source_col: Text column name.
        category_col: Category column name.

    Returns:
        DataFrame with per-category stats for the filtered articles.
    """
    mask = df[source_col].str.lower().str.contains(topic_keyword.lower(), na=False)
    filtered = df[mask].copy()

    if len(filtered) == 0:
        logger.warning(f"No articles found containing '{topic_keyword}'")
        return pd.DataFrame()

    if "sentiment_compound" not in filtered.columns:
        sents = filtered[source_col].apply(analyze_sentiment)
        filtered["sentiment_compound"] = sents.apply(lambda r: r["compound"])
        filtered["sentiment_label"]    = sents.apply(lambda r: r["sentiment_label"])

    summary = (
        filtered.groupby(category_col)
        .agg(
            article_count=("sentiment_compound", "count"),
            avg_sentiment=("sentiment_compound", "mean"),
        )
        .sort_values("article_count", ascending=False)
    )
    return summary
