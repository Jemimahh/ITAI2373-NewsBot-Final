"""
src/language_models/generator.py
Content enhancement and insight generation via ollama — Module B.2 / B.4.
"""

import json
import logging
from src.language_models.summarizer import _call_ollama

logger = logging.getLogger(__name__)


def enhance_content(
    article_text: str,
    category: str = None,
    entities: dict = None,
) -> dict:
    """
    Expand article analysis with three layers of contextual enrichment.

    Layers:
        1. background_context — historical or domain context
        2. related_trends     — current trends relevant to article
        3. implications       — potential consequences or significance

    Args:
        article_text: Raw or cleaned article text.
        category: News category hint (e.g. "tech").
        entities: Pre-computed entity dict from NER (optional).

    Returns:
        Dict with keys: background_context, related_trends, implications,
                        entities_to_watch (list).
    """
    entity_hint = ""
    if entities:
        all_ents = [e for lst in entities.values() for e in lst[:3]]
        if all_ents:
            entity_hint = f" Key entities: {', '.join(all_ents[:5])}."

    cat_hint = f" Category: {category}." if category else ""
    system   = (
        "You are a senior news analyst. Respond ONLY with valid JSON. "
        "No markdown, no preamble, no explanation outside the JSON object."
    )
    prompt = (
        f"Analyze this news article and return a JSON object with these exact keys:\n"
        f"  background_context (string: 2-3 sentences of relevant background)\n"
        f"  related_trends     (string: 2-3 sentences on broader trends)\n"
        f"  implications       (string: 2-3 sentences on potential impact)\n"
        f"  entities_to_watch  (list of 3-5 entity name strings)\n\n"
        f"Article:{cat_hint}{entity_hint}\n{article_text[:2500]}"
    )

    raw = _call_ollama(prompt, system)

    try:
        clean = raw.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("enhance_content: JSON parse failed, returning raw text.")
        return {"raw_response": raw}


def generate_insights(
    article_text: str,
    nlp_metadata: dict = None,
) -> dict:
    """
    Generate structured insights from an article and its NLP metadata.

    Grounds the LLM analysis in pre-computed quantitative signals
    (sentiment, TF-IDF, entities) to reduce hallucination.

    Args:
        article_text: Raw article text.
        nlp_metadata: Dict with optional keys:
            - sentiment_label (str)
            - sentiment_compound (float)
            - top_tfidf (list of [term, score] pairs)
            - entities (dict)
            - lda_topic_label (str)
            - nmf_topic_label (str)

    Returns:
        Dict with: key_findings, patterns, entities_of_interest,
                   sentiment_drivers, anomalies, recommended_queries.
    """
    meta_str = ""
    if nlp_metadata:
        parts = []
        if "sentiment_label" in nlp_metadata:
            parts.append(f"Sentiment: {nlp_metadata['sentiment_label']} "
                         f"(score: {nlp_metadata.get('sentiment_compound', 'N/A')})")
        if "top_tfidf" in nlp_metadata:
            terms = [t for t, _ in nlp_metadata["top_tfidf"][:5]]
            parts.append(f"Top TF-IDF terms: {', '.join(terms)}")
        if "lda_topic_label" in nlp_metadata:
            parts.append(f"Topic (LDA): {nlp_metadata['lda_topic_label']}")
        if "nmf_topic_label" in nlp_metadata:
            parts.append(f"Topic (NMF): {nlp_metadata['nmf_topic_label']}")
        meta_str = "\n".join(parts)

    system = (
        "You are an expert NLP analyst. Respond ONLY with valid JSON. "
        "Ground all observations in the provided article and metadata."
    )
    prompt = (
        "Analyze this news article and return a JSON object with these exact keys:\n"
        "  key_findings         (list of 3-5 finding strings)\n"
        "  patterns             (list of 2-3 pattern strings)\n"
        "  entities_of_interest (list of entity name strings)\n"
        "  sentiment_drivers    (list of 2-3 phrases driving sentiment)\n"
        "  anomalies            (list of unusual or noteworthy elements)\n"
        "  recommended_queries  (list of 4 natural language questions a reader might ask)\n\n"
        f"Pre-computed NLP metadata:\n{meta_str}\n\n"
        f"Article:\n{article_text[:2500]}"
    )

    raw = _call_ollama(prompt, system)

    try:
        clean = raw.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("generate_insights: JSON parse failed.")
        return {"raw_response": raw}
