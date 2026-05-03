"""
settings.py
-----------
Centralized configuration for NewsBot Intelligence System 2.0.

All tunable parameters live here. Import this module anywhere in the
project rather than hardcoding values in source files.

Usage:
    from config.settings import DATA, MODEL_B, ANALYSIS
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = DATA_DIR / "models"
RESULTS_DIR   = DATA_DIR / "results"
REPORTS_DIR   = ROOT / "reports"


# ── Dataset ───────────────────────────────────────────────────────────────────

DATA = {
    "bbc_path":      RAW_DIR / "bbc",
    "categories":    ["business", "entertainment", "politics", "sport", "tech"],
    "test_split":    0.2,
    "random_state":  42,
    "encoding":      "utf-8",
}


# ── Text Preprocessing ────────────────────────────────────────────────────────

PREPROCESSING = {
    "min_token_length": 2,
    "max_doc_length":   512,
    "remove_stopwords": True,
    "lemmatize":        True,
    "language":         "english",
}


# ── Module A: Analysis ────────────────────────────────────────────────────────

ANALYSIS = {
    # Topic modeling
    "n_topics":        10,
    "lda_passes":      15,
    "lda_iterations":  100,
    "nmf_max_iter":    200,

    # Classification
    "classifier":      "svm",          # "svm" | "logreg" | "rf"
    "tfidf_max_features": 10000,

    # NER
    "spacy_model":     "en_core_web_sm",
}


# ── Module B: Language Models ─────────────────────────────────────────────────

MODEL_B = {
    # ollama model tag — change to "llama3.2:1b" for faster CPU inference
    "llm_model":         "llama3.2",
    "llm_temperature":   0.2,

    # sentence-transformers model for embeddings
    # "fast"         → all-MiniLM-L6-v2      (384-dim, great for CPU)
    # "balanced"     → all-mpnet-base-v2     (768-dim, better quality)
    # "multilingual" → paraphrase-multilingual-MiniLM-L12-v2
    "embedding_model":   "all-MiniLM-L6-v2",
    "embedding_batch":   32,

    "summary_style":     "standard",    # "brief" | "standard" | "detailed" | "formal"
    "max_insights":      10,            # max articles fed to insight generator
}


# ── Module C: Multilingual ────────────────────────────────────────────────────

MULTILINGUAL = {
    "target_language":   "en",
    "supported_languages": ["en", "es", "fr", "de", "pt", "ar"],
    "translation_service": "deep_translator",   # free, no API key
}


# ── API Keys ──────────────────────────────────────────────────────────────────
# Loaded from config/api_keys.txt if present.
# All keys are optional — the system runs fully without them.

_keys_file = Path(__file__).parent / "api_keys.txt"
API_KEYS: dict[str, str] = {}

if _keys_file.exists():
    for line in _keys_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            API_KEYS[key.strip()] = value.strip()


# ── Logging ───────────────────────────────────────────────────────────────────

LOGGING = {
    "level":  "INFO",
    "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
}
