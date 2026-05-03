"""
src/analysis/ner_extractor.py
Named Entity Recognition — extended from ITAI 2373 midterm Module 8.

Provides entity extraction, frequency analysis, entity relationship
mapping (co-occurrence), and entity-level sentiment tagging.
"""

import logging
from collections import Counter
import pandas as pd
from config.settings import SPACY_MODEL

logger = logging.getLogger(__name__)

ENTITY_TYPES_OF_INTEREST = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "LAW"]


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities from text using spaCy.

    Args:
        text: Raw article text.

    Returns:
        Dict mapping entity_type → [entity_strings].
    """
    from src.data_processing.text_preprocessor import get_nlp
    nlp = get_nlp()
    doc = nlp(text[:10_000])

    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES_OF_INTEREST:
            continue
        surface = ent.text.strip()
        if not surface or len(surface) < 2:
            continue
        entities.setdefault(ent.label_, []).append(surface)
    return entities


def get_entity_frequencies(df: pd.DataFrame, top_n: int = 20) -> Counter:
    """
    Count entity frequency across an entire DataFrame.

    Args:
        df: DataFrame with 'entities' column (dict of type → list).
        top_n: Number of top entities to return.

    Returns:
        Counter of entity_text → count.
    """
    freq: Counter = Counter()
    for entities_dict in df["entities"]:
        for entity_list in entities_dict.values():
            freq.update(entity_list)
    return freq.most_common(top_n)


def build_entity_cooccurrence(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """
    Build an entity co-occurrence matrix.

    Two entities co-occur if they appear in the same article.
    Used to identify relationship networks between people, orgs, etc.

    Args:
        df: DataFrame with 'entities' column.
        min_count: Minimum co-occurrence count to include in output.

    Returns:
        DataFrame of (entity_a, entity_b, cooccurrence_count).
    """
    from itertools import combinations
    cooc: Counter = Counter()

    for entities_dict in df["entities"]:
        all_ents = []
        for ent_list in entities_dict.values():
            all_ents.extend(set(ent_list))  # dedupe within type
        for a, b in combinations(sorted(set(all_ents)), 2):
            cooc[(a, b)] += 1

    rows = [{"entity_a": a, "entity_b": b, "count": cnt}
            for (a, b), cnt in cooc.items() if cnt >= min_count]
    result = pd.DataFrame(rows).sort_values("count", ascending=False)
    logger.info(f"Entity co-occurrence pairs: {len(result)}")
    return result


def entity_sentiment_profile(
    df: pd.DataFrame,
    entity_name: str,
    entity_type: str = None,
) -> dict:
    """
    Compute sentiment statistics for articles mentioning a specific entity.

    Args:
        df: DataFrame with 'entities' and 'sentiment_compound' columns.
        entity_name: Entity to analyze (e.g. "Apple", "Boris Johnson").
        entity_type: Optional entity type filter (e.g. "ORG", "PERSON").

    Returns:
        Dict with count, mean_sentiment, label_distribution.
    """
    def mentions(row):
        if entity_type:
            ents = row["entities"].get(entity_type, [])
        else:
            ents = [e for lst in row["entities"].values() for e in lst]
        return entity_name.lower() in [e.lower() for e in ents]

    mask = df.apply(mentions, axis=1)
    sub  = df[mask]

    if len(sub) == 0:
        return {"count": 0, "mean_sentiment": None, "label_distribution": {}}

    return {
        "count":               len(sub),
        "mean_sentiment":      float(sub["sentiment_compound"].mean()),
        "label_distribution":  sub["sentiment_label"].value_counts().to_dict(),
        "categories":          sub["category"].value_counts().to_dict(),
    }
