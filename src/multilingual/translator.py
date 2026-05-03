"""
src/multilingual/translator.py
Translation services via deep-translator (Google Translate API wrapper).
Module C — Multilingual Intelligence.
"""

import logging
from config.settings import SUPPORTED_LANGUAGES, DEFAULT_TARGET_LANG, TRANSLATION_SERVICE

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 4500  # Google Translate per-request limit


def translate_text(
    text: str,
    target_lang: str = DEFAULT_TARGET_LANG,
    source_lang: str = "auto",
) -> dict:
    """
    Translate text to a target language.

    Uses GoogleTranslator from deep-translator (no API key required).

    Args:
        text: Input text to translate.
        target_lang: ISO 639-1 language code (e.g. "en", "fr", "de").
        source_lang: Source language code or "auto" for auto-detection.

    Returns:
        Dict with: translated_text, source_lang, target_lang, char_count.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise RuntimeError("deep-translator not installed. Run: pip install deep-translator")

    # Chunk long texts
    chunks = _chunk_text(str(text), MAX_CHUNK_CHARS)
    translated_chunks = []

    for chunk in chunks:
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_chunks.append(translator.translate(chunk))
        except Exception as e:
            logger.warning(f"Translation chunk failed: {e}")
            translated_chunks.append(chunk)  # fallback: keep original

    translated = " ".join(translated_chunks)

    return {
        "translated_text": translated,
        "source_lang":     source_lang,
        "target_lang":     target_lang,
        "char_count":      len(translated),
        "n_chunks":        len(chunks),
    }


def translate_article_batch(
    df,
    text_col: str = "cleaned_text",
    target_lang: str = DEFAULT_TARGET_LANG,
    n_samples: int = None,
) -> list[dict]:
    """
    Translate multiple articles from a DataFrame.

    Args:
        df: DataFrame with text column.
        text_col: Column to translate.
        target_lang: Target language code.
        n_samples: Optional limit on number of articles to translate.

    Returns:
        List of translation result dicts.
    """
    from tqdm import tqdm

    subset = df if n_samples is None else df.head(n_samples)
    results = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Translating → {target_lang}"):
        try:
            result = translate_text(row[text_col], target_lang=target_lang)
            result["category"] = row.get("category")
            results.append(result)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            results.append({"translated_text": None, "error": str(e)})

    return results


def get_supported_languages() -> dict:
    """Return the supported language code → name mapping."""
    return SUPPORTED_LANGUAGES.copy()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars characters at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks, current = [], ""
    for sentence in text.split(". "):
        if len(current) + len(sentence) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current += ". " + sentence if current else sentence

    if current:
        chunks.append(current.strip())
    return chunks
