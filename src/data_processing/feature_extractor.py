"""
src/data_processing/feature_extractor.py
TF-IDF vectorization and custom NLP feature extraction.

Provides both sklearn TF-IDF wrappers and the custom TF-IDF
implementation carried over from the ITAI 2373 midterm.
"""

import numpy as np
import logging
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from config.settings import (
    TFIDF_MAX_FEATURES, TFIDF_MAX_DF, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, EXTRA_STOPWORDS
)

logger = logging.getLogger(__name__)


# ── Sklearn wrappers ──────────────────────────────────────────────────────

def build_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Return a configured TF-IDF vectorizer.

    Merges default settings from config with any overrides in kwargs.
    """
    defaults = dict(
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    defaults.update(kwargs)
    return TfidfVectorizer(**defaults)


def build_count_vectorizer(**kwargs) -> CountVectorizer:
    """Return a configured Count (BoW) vectorizer for LDA."""
    defaults = dict(
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    defaults.update(kwargs)
    return CountVectorizer(**defaults)


def fit_vectorizers(texts: list[str]):
    """
    Fit both TF-IDF and Count vectorizers on a corpus.

    Returns:
        Tuple of (tfidf_matrix, count_matrix, tfidf_vec, count_vec, tfidf_vocab, count_vocab)
    """
    tfidf_vec = build_tfidf_vectorizer()
    count_vec = build_count_vectorizer()

    tfidf_matrix = tfidf_vec.fit_transform(texts)
    count_matrix = count_vec.fit_transform(texts)

    tfidf_vocab = tfidf_vec.get_feature_names_out()
    count_vocab = count_vec.get_feature_names_out()

    logger.info(f"TF-IDF matrix: {tfidf_matrix.shape}, Count matrix: {count_matrix.shape}")
    return tfidf_matrix, count_matrix, tfidf_vec, count_vec, tfidf_vocab, count_vocab


# ── Custom TF-IDF (midterm implementation) ────────────────────────────────

def build_global_vocab_and_idf(documents: list[list[str]]) -> tuple[dict, dict]:
    """
    Build vocabulary and IDF scores from a list of tokenized documents.

    IDF formula: log((N + 1) / (df + 1)) + 1  (sklearn-style smoothed)

    Args:
        documents: List of token lists (output of tokenize_and_process).

    Returns:
        Tuple of (vocab dict token→idx, idf dict token→score).
    """
    from collections import Counter
    import math

    N = len(documents)
    df_counts: dict[str, int] = {}

    all_tokens: set[str] = set()
    for doc in documents:
        unique = set(doc)
        all_tokens.update(unique)
        for token in unique:
            df_counts[token] = df_counts.get(token, 0) + 1

    vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    idf   = {token: math.log((N + 1) / (df_counts.get(token, 0) + 1)) + 1
             for token in vocab}

    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab, idf


def calculate_document_tfidf_vector(
    tokens: list[str],
    vocab: dict,
    idf: dict,
) -> np.ndarray:
    """
    Compute a TF-IDF vector for a single tokenized document.

    Args:
        tokens: List of tokens for one document.
        vocab: Token → index mapping from build_global_vocab_and_idf.
        idf: Token → IDF score mapping.

    Returns:
        Numpy array of shape (vocab_size,) with TF-IDF weights.
    """
    from collections import Counter

    tf_counts = Counter(tokens)
    total = len(tokens) if tokens else 1

    vec = np.zeros(len(vocab))
    for token, count in tf_counts.items():
        if token in vocab:
            tf  = count / total
            vec[vocab[token]] = tf * idf.get(token, 1.0)

    return vec


def get_top_tfidf_terms(
    tokens: list[str],
    vocab: dict,
    idf: dict,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-N TF-IDF terms for a document.

    Args:
        tokens: Token list.
        vocab: Vocabulary mapping.
        idf: IDF mapping.
        top_n: Number of top terms to return.

    Returns:
        List of (term, score) tuples sorted by descending score.
    """
    vec = calculate_document_tfidf_vector(tokens, vocab, idf)
    top_idx = np.argsort(vec)[::-1][:top_n]
    idx_to_token = {v: k for k, v in vocab.items()}
    return [(idx_to_token[i], float(vec[i])) for i in top_idx if vec[i] > 0]


# ── Custom NLP features (midterm Module 7 features) ───────────────────────

def extract_custom_features(text: str) -> dict:
    """
    Extract numeric NLP features used in classification.

    Features:
        - avg_sentence_length
        - avg_dependency_tree_depth
        - noun_phrase_count
        - verb_phrase_count
        - passive_voice_instance_count

    Args:
        text: Raw article text.

    Returns:
        Dict of feature name → float value.
    """
    from src.data_processing.text_preprocessor import get_nlp

    nlp = get_nlp()
    doc = nlp(text[:10_000])

    sents = list(doc.sents)
    n_sents = len(sents) if sents else 1

    avg_sent_len = sum(len(s) for s in sents) / n_sents

    def tree_depth(token):
        depth = 0
        while token.head != token:
            token = token.head
            depth += 1
        return depth

    depths = [tree_depth(t) for t in doc if not t.is_space]
    avg_dep_depth = sum(depths) / len(depths) if depths else 0

    noun_phrases = len(list(doc.noun_chunks))

    verb_phrases = sum(1 for t in doc if t.pos_ == "VERB")

    passive_count = sum(
        1 for t in doc
        if t.dep_ == "auxpass" and t.head.pos_ == "VERB" and t.head.tag_ == "VBN"
    )

    return {
        "avg_sentence_length":       avg_sent_len,
        "avg_dependency_tree_depth": avg_dep_depth,
        "noun_phrase_count":         noun_phrases,
        "verb_phrase_count":         verb_phrases,
        "passive_voice_instance_count": passive_count,
    }
