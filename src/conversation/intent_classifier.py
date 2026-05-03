"""
src/conversation/intent_classifier.py
Intent classification for the NewsBot conversational interface.
Hybrid approach: rule-based patterns + TF-IDF similarity fallback.
"""

import logging
import numpy as np
from src.conversation.query_processor import INTENT_PATTERNS, QueryProcessor

logger = logging.getLogger(__name__)

INTENT_EXAMPLES = {
    "filter_by_sentiment":  ["show positive news", "find negative articles", "neutral coverage"],
    "filter_by_category":   ["tech news", "sports articles", "business coverage", "political stories"],
    "summarize":            ["summarize this", "give me a brief overview", "tldr"],
    "top_entities":         ["who is mentioned most", "top organizations", "most frequent people"],
    "topic_query":          ["what topics are covered", "main themes", "what is this about"],
    "search":               ["find articles about AI", "show me articles on Brexit"],
    "stats":                ["how many tech articles", "sentiment breakdown", "category statistics"],
    "comparison":           ["compare tech vs business", "difference between categories"],
    "general":              ["hello", "what can you do", "help me"],
}


class IntentClassifier:
    """
    Classify user query intent using rules + optional embedding fallback.

    Usage::

        clf = IntentClassifier()
        intent, confidence = clf.classify("show me positive tech news")
    """

    def __init__(self, use_embeddings: bool = False):
        """
        Args:
            use_embeddings: If True, use sentence embeddings as fallback.
        """
        self.use_embeddings = use_embeddings
        self._qp = QueryProcessor()
        self._intent_embs = None

    def classify(self, query: str) -> tuple[str, float]:
        """
        Classify query intent with confidence score.

        Args:
            query: User query string.

        Returns:
            Tuple of (intent_string, confidence_0_to_1).
        """
        # Rule-based: high confidence
        rule_intent = self._qp.classify_intent(query)
        if rule_intent != "general":
            return rule_intent, 0.95

        # Embedding fallback
        if self.use_embeddings:
            emb_intent, emb_conf = self._embedding_classify(query)
            return emb_intent, emb_conf

        return "general", 0.5

    def _embedding_classify(self, query: str) -> tuple[str, float]:
        """Use sentence embeddings to find closest intent example."""
        try:
            from sentence_transformers import SentenceTransformer
            if self._intent_embs is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                all_examples = []
                self._intent_keys = []
                for intent, examples in INTENT_EXAMPLES.items():
                    for ex in examples:
                        all_examples.append(ex)
                        self._intent_keys.append(intent)
                self._intent_embs = model.encode(all_examples, normalize_embeddings=True)
                self._emb_model = model

            q_emb  = self._emb_model.encode([query], normalize_embeddings=True)
            scores = (self._intent_embs @ q_emb.T).flatten()
            best   = int(np.argmax(scores))
            return self._intent_keys[best], float(scores[best])
        except Exception as e:
            logger.warning(f"Embedding classify failed: {e}")
            return "general", 0.5
