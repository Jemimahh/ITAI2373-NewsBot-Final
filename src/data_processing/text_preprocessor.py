"""
src/data_processing/text_preprocessor.py
Enhanced text preprocessing pipeline — extended from ITAI 2373 midterm.

Provides clean_text, expand_contractions, tokenize_and_process,
and extract_named_entities using spaCy en_core_web_sm.
"""

import re
import logging
from typing import Optional
import spacy
import contractions as contractions_lib
from config.settings import SPACY_MODEL, EXTRA_STOPWORDS, MIN_TOKEN_LENGTH

logger = logging.getLogger(__name__)


def load_nlp_model(model_name: str = SPACY_MODEL) -> spacy.Language:
    """Load and return spaCy language model. Downloads if not found."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model: {model_name}")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)


# Lazy-load the model once at module level
_nlp: Optional[spacy.Language] = None


def get_nlp() -> spacy.Language:
    """Return singleton spaCy model instance."""
    global _nlp
    if _nlp is None:
        _nlp = load_nlp_model()
    return _nlp


def expand_contractions(text: str) -> str:
    """
    Expand English contractions.

    Args:
        text: Raw input string.

    Returns:
        Text with contractions expanded (e.g. "don't" → "do not").
    """
    try:
        return contractions_lib.fix(text)
    except Exception:
        return text


def clean_text(text: str) -> str:
    """
    Apply baseline text cleaning pipeline.

    Steps:
        1. Expand contractions
        2. Lowercase
        3. Remove URLs, email addresses, HTML tags
        4. Remove non-alphabetic characters (keep spaces)
        5. Collapse multiple whitespace

    Args:
        text: Raw article text.

    Returns:
        Cleaned lowercase string.
    """
    if not isinstance(text, str):
        text = str(text)

    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                   # emails
    text = re.sub(r"<[^>]+>", " ", text)                   # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                  # non-alpha
    text = re.sub(r"\s+", " ", text).strip()               # whitespace
    return text


def tokenize_and_process(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    remove_extra_stops: bool = True,
) -> list[str]:
    """
    Tokenize text using spaCy and apply optional processing.

    Args:
        text: Cleaned input text.
        remove_stopwords: If True, remove spaCy stopwords.
        lemmatize: If True, return lemmas instead of surface forms.
        remove_extra_stops: If True, also remove EXTRA_STOPWORDS from settings.

    Returns:
        List of processed tokens.
    """
    nlp = get_nlp()
    doc = nlp(text)

    extra = set(EXTRA_STOPWORDS) if remove_extra_stops else set()

    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if remove_stopwords and token.is_stop:
            continue
        form = token.lemma_ if lemmatize else token.text
        if len(form) < MIN_TOKEN_LENGTH:
            continue
        if form in extra:
            continue
        tokens.append(form.lower())

    return tokens


def extract_named_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities grouped by entity type.

    Entity types (spaCy en_core_web_sm):
        PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, PRODUCT, EVENT, LAW

    Args:
        text: Raw or cleaned article text.

    Returns:
        Dict mapping entity type → list of entity strings.
    """
    nlp = get_nlp()
    doc = nlp(text[:10_000])  # cap for performance

    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = ent.label_
        surface = ent.text.strip()
        if not surface:
            continue
        entities.setdefault(label, []).append(surface)

    return entities


def get_pos_tags(text: str) -> list[tuple[str, str]]:
    """
    Return (token, POS_tag) pairs for a text.

    Args:
        text: Input text.

    Returns:
        List of (token_text, universal_POS) tuples.
    """
    nlp = get_nlp()
    doc = nlp(text[:10_000])
    return [(token.text, token.pos_) for token in doc if not token.is_space]


def preprocess_dataframe(df, text_col: str = "text", label_col: str = "category"):
    """
    Apply full preprocessing pipeline to a DataFrame.

    Creates new columns:
        - cleaned_text
        - tokens
        - entities
        - pos_tags

    Args:
        df: Input DataFrame with text and label columns.
        text_col: Name of the column containing article text.
        label_col: Name of the column containing category labels.

    Returns:
        DataFrame with new preprocessing columns added.
    """
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas(desc="Preprocessing")

    df = df.copy()
    df["cleaned_text"] = df[text_col].progress_apply(clean_text)
    df["tokens"]       = df["cleaned_text"].progress_apply(tokenize_and_process)
    df["entities"]     = df[text_col].progress_apply(extract_named_entities)
    df["pos_tags"]     = df[text_col].progress_apply(get_pos_tags)

    logger.info(f"Preprocessing complete: {len(df)} documents.")
    return df
