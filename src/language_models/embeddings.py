"""
src/language_models/embeddings.py
Semantic embeddings for similarity search and query expansion.
Uses sentence-transformers (all-MiniLM-L6-v2) as the default model.
"""

import logging
import numpy as np
from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticSearchEngine:
    """
    Lightweight semantic search over a news article corpus.

    Usage::

        engine = SemanticSearchEngine()
        engine.index(df, text_col='cleaned_text')
        results = engine.search("interest rate decision", top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Args:
            model_name: sentence-transformers model identifier.
        """
        self.model_name = model_name
        self._model     = None
        self._embeddings: np.ndarray = None
        self._df        = None
        self._is_indexed = False

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    def index(self, df, text_col: str = "cleaned_text", batch_size: int = 32):
        """
        Encode all documents and build the search index.

        Args:
            df: DataFrame with article text column.
            text_col: Column containing text to encode.
            batch_size: Encoding batch size.
        """
        self._load_model()
        self._df = df.reset_index(drop=True)
        texts    = self._df[text_col].tolist()

        logger.info(f"Encoding {len(texts)} documents...")
        self._embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._is_indexed = True
        logger.info("Indexing complete.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top-k most semantically similar articles to a query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: rank, score, category, text_preview.
        """
        self._check_indexed()
        self._load_model()

        q_emb  = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            row = self._df.iloc[idx]
            results.append({
                "rank":         rank,
                "score":        float(scores[idx]),
                "category":     row.get("category", "unknown"),
                "text_preview": row.get("cleaned_text", "")[:200] + "...",
                "index":        int(idx),
            })
        return results

    def expand_query(self, query: str, n_terms: int = 5, vocab: list[str] = None) -> list[str]:
        """
        Expand a query with semantically related terms from vocabulary.

        Args:
            query: Original query string.
            n_terms: Number of additional terms to suggest.
            vocab: Optional vocabulary list. Defaults to a small seed vocab.

        Returns:
            List of semantically related terms.
        """
        self._load_model()

        if vocab is None:
            vocab = [
                "economy", "technology", "government", "sports", "entertainment",
                "finance", "election", "climate", "innovation", "trade", "policy",
                "market", "research", "championship", "legislation", "startup",
                "investment", "competition", "regulation", "announcement"
            ]

        q_emb   = self._model.encode([query], normalize_embeddings=True)
        v_emb   = self._model.encode(vocab, normalize_embeddings=True)
        scores  = (v_emb @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:n_terms]
        return [vocab[i] for i in top_idx]

    def _check_indexed(self):
        if not self._is_indexed:
            raise RuntimeError("Call .index(df) before searching.")
