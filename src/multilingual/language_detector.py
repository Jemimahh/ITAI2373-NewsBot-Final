"""
src/multilingual/language_detector.py
Language identification using langdetect with confidence scoring.
Module C — Multilingual Intelligence.
"""

import logging
from config.settings import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


def detect_language(text: str) -> dict:
    """
    Detect the language of a text string.

    Args:
        text: Input text (min ~20 chars for reliable detection).

    Returns:
        Dict with: language_code, language_name, confidence.
    """
    try:
        from langdetect import detect, detect_langs
        from langdetect import DetectorFactory
        DetectorFactory.seed = 42  # reproducibility

        lang_probs = detect_langs(str(text)[:500])
        top = lang_probs[0]
        code = str(top.lang)
        name = SUPPORTED_LANGUAGES.get(code, code.upper())

        return {
            "language_code":  code,
            "language_name":  name,
            "confidence":     round(top.prob, 4),
            "all_detections": [(str(lp.lang), round(lp.prob, 4)) for lp in lang_probs],
        }
    except ImportError:
        raise RuntimeError("langdetect not installed. Run: pip install langdetect")
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return {"language_code": "unknown", "language_name": "Unknown", "confidence": 0.0}


def detect_dataframe(df, text_col: str = "cleaned_text") -> "pd.DataFrame":
    """
    Apply language detection across a DataFrame.

    Adds columns: detected_language, language_confidence.

    Args:
        df: DataFrame with text column.
        text_col: Column to detect language on.

    Returns:
        DataFrame with new language columns.
    """
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas(desc="Detecting language")

    df = df.copy()
    results = df[text_col].progress_apply(detect_language)
    df["detected_language"]    = results.apply(lambda r: r["language_code"])
    df["language_confidence"]  = results.apply(lambda r: r["confidence"])
    df["language_name"]        = results.apply(lambda r: r["language_name"])
    return df


def is_english(text: str, threshold: float = 0.85) -> bool:
    """
    Return True if text is detected as English with sufficient confidence.

    Args:
        text: Input text.
        threshold: Minimum confidence to classify as English.

    Returns:
        Boolean.
    """
    result = detect_language(text)
    return result["language_code"] == "en" and result["confidence"] >= threshold
