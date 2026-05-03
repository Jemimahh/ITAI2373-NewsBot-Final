"""
src/utils/evaluation.py
Model evaluation utilities — classification metrics, topic coherence, clustering scores.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, silhouette_score
)
from config.settings import CATEGORIES

logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with accuracy, macro/weighted precision, recall, F1.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0)

    return {
        "accuracy":           round(acc, 4),
        "macro_precision":    round(prec_macro, 4),
        "macro_recall":       round(rec_macro, 4),
        "macro_f1":           round(f1_macro, 4),
        "weighted_precision": round(prec_wt, 4),
        "weighted_recall":    round(rec_wt, 4),
        "weighted_f1":        round(f1_wt, 4),
        "report":             report,
        "confusion_matrix":   confusion_matrix(y_true, y_pred, labels=CATEGORIES),
    }


def clustering_evaluation(X, labels) -> dict:
    """
    Evaluate clustering quality.

    Args:
        X: Feature matrix used for clustering.
        labels: Cluster label array.

    Returns:
        Dict with silhouette score and cluster size stats.
    """
    from collections import Counter
    sil  = silhouette_score(X, labels, sample_size=min(1000, len(X)))
    dist = Counter(labels)
    return {
        "silhouette_score": round(float(sil), 4),
        "n_clusters":       len(dist),
        "cluster_sizes":    dict(sorted(dist.items())),
        "largest_cluster":  max(dist.values()),
        "smallest_cluster": min(dist.values()),
    }


def topic_coherence_proxy(topics: dict, df, text_col: str = "cleaned_text", top_n: int = 10) -> dict:
    """
    Compute a proxy topic coherence score (PMI-based).

    Uses co-occurrence in the corpus as a coherence signal.
    Note: For production use, consider gensim's CoherenceModel (C_v metric).

    Args:
        topics: Dict of topic_id → [(word, weight), ...].
        df: DataFrame with cleaned text.
        text_col: Column with text.
        top_n: Number of top words per topic to evaluate.

    Returns:
        Dict of topic_id → mean co-occurrence score.
    """
    from itertools import combinations

    texts = df[text_col].tolist()
    scores = {}

    for t_id, words in topics.items():
        top_words = [w for w, _ in words[:top_n]]
        pairs = list(combinations(top_words, 2))
        if not pairs:
            scores[t_id] = 0.0
            continue

        pair_scores = []
        for w1, w2 in pairs:
            docs_with_both = sum(1 for t in texts if w1 in t and w2 in t)
            docs_with_w1   = sum(1 for t in texts if w1 in t) or 1
            docs_with_w2   = sum(1 for t in texts if w2 in t) or 1
            pmi = np.log((docs_with_both * len(texts) + 1) / (docs_with_w1 * docs_with_w2 + 1))
            pair_scores.append(pmi)

        scores[t_id] = round(float(np.mean(pair_scores)), 4)

    return scores
