"""
config/settings.py
NewsBot Intelligence System 2.0 — Central Configuration
ITAI 2373 Final Project | Houston Community College
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (never commit .env)
load_dotenv()

# ── Project Paths ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = DATA_DIR / "models"
RESULTS_DIR    = DATA_DIR / "results"
NOTEBOOKS_DIR  = ROOT_DIR / "notebooks"
DOCS_DIR       = ROOT_DIR / "docs"

# ── Dataset ───────────────────────────────────────────────────────────────
DATASET_NAME    = "bbc-news-data.csv"
DATASET_PATH    = RAW_DATA_DIR / DATASET_NAME
TEXT_COLUMN     = "text"
LABEL_COLUMN    = "category"
CATEGORIES      = ["tech", "business", "politics", "sport", "entertainment"]
RANDOM_STATE    = 42

# ── Preprocessing ─────────────────────────────────────────────────────────
SPACY_MODEL         = "en_core_web_sm"
MAX_TEXT_LENGTH     = 10_000   # characters
MIN_TOKEN_LENGTH    = 2
EXTRA_STOPWORDS     = [
    "said", "would", "could", "also", "one", "two", "three", "new",
    "year", "years", "time", "like", "make", "made", "says", "say",
    "mr", "mrs", "ms", "people", "way", "first", "last", "week",
    "month", "day", "told", "come", "go", "going", "get", "got",
    "us", "uk", "back", "may", "will", "take", "use", "used"
]

# ── TF-IDF ────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 5000
TFIDF_MAX_DF        = 0.90
TFIDF_MIN_DF        = 5
TFIDF_NGRAM_RANGE   = (1, 2)

# ── Topic Modeling ────────────────────────────────────────────────────────
N_TOPICS            = 10
LDA_MAX_ITER        = 25
LDA_DOC_PRIOR       = 0.1
LDA_WORD_PRIOR      = 0.01
NMF_MAX_ITER        = 300
NMF_ALPHA           = 0.1
NMF_L1_RATIO        = 0.5
CLUSTER_K_RANGE     = range(2, 13)

# ── LLM (ollama / Module B) ───────────────────────────────────────────────
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
LLM_TEMPERATURE     = 0.3
LLM_MAX_TOKENS      = 512
LLM_TIMEOUT_SECONDS = 60

# ── Multilingual (Module C) ───────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
}
DEFAULT_TARGET_LANG = "en"
TRANSLATION_SERVICE = "google"   # Options: "google", "deepl" (requires API key)

# ── Sentiment ─────────────────────────────────────────────────────────────
VADER_POSITIVE_THRESHOLD  =  0.05
VADER_NEGATIVE_THRESHOLD  = -0.05

# ── Visualization ─────────────────────────────────────────────────────────
PLOT_STYLE      = "dark"
PLOT_DPI        = 120
PLOT_FACECOLOR  = "#0f1117"
PLOT_PALETTE    = {
    "tech":          "#4fc3f7",
    "business":      "#81c784",
    "politics":      "#e57373",
    "sport":         "#ffb74d",
    "entertainment": "#ce93d8",
}

# ── API Keys (loaded from environment — never hardcode) ───────────────────
DEEPL_API_KEY       = os.getenv("DEEPL_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")  # for web frontend only
KAGGLE_USERNAME     = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY          = os.getenv("KAGGLE_KEY", "")

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT  = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def validate_paths():
    """Create required directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return True


if __name__ == "__main__":
    validate_paths()
    print(f"Project root : {ROOT_DIR}")
    print(f"Dataset path : {DATASET_PATH}")
    print(f"ollama model : {OLLAMA_MODEL}")
    print(f"LLM host     : {OLLAMA_HOST}")
