"""
src/conversation/query_processor.py
Natural language query handling for the NewsBot conversational interface.
Module D — Conversational Interface.
"""

import logging
import re
from config.settings import CATEGORIES

logger = logging.getLogger(__name__)

INTENT_PATTERNS = {
    "filter_by_sentiment": [
        r"positive\s+news", r"negative\s+news", r"neutral\s+news",
        r"optimistic", r"pessimistic",
    ],
    "filter_by_category": [
        rf"\b{cat}\b" for cat in CATEGORIES
    ],
    "summarize": [
        r"summar", r"brief", r"tldr", r"overview", r"gist",
    ],
    "top_entities": [
        r"(who|what|which)\s+(people|organizations|companies|countries)",
        r"most\s+mentioned", r"top\s+entities",
    ],
    "topic_query": [
        r"topic", r"theme", r"about\s+what", r"main\s+subject",
    ],
    "search": [
        r"find\s+articles?\s+about", r"search\s+for", r"show\s+me\s+articles?",
        r"articles?\s+(about|on|covering)",
    ],
    "stats": [
        r"how\s+many", r"count", r"statistics", r"breakdown", r"distribution",
    ],
    "comparison": [
        r"compar", r"vs\.?", r"versus", r"difference\s+between",
    ],
}


class QueryProcessor:
    """
    Natural language query processor for the NewsBot system.

    Parses user queries into structured intents with extracted
    parameters (category, sentiment, keyword, etc.).

    Usage::

        qp = QueryProcessor(df)
        result = qp.process("Show me positive tech news from this week")
        # result = {intent, filters, response_data, response_text}
    """

    def __init__(self, df=None):
        """
        Args:
            df: DataFrame with article data for query execution.
        """
        self.df = df

    def classify_intent(self, query: str) -> str:
        """
        Classify the primary intent of a natural language query.

        Args:
            query: User query string.

        Returns:
            Intent string (one of INTENT_PATTERNS keys or 'general').
        """
        q_lower = query.lower()
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    return intent
        return "general"

    def extract_filters(self, query: str) -> dict:
        """
        Extract filter parameters from a query string.

        Extracts: category, sentiment_label, keyword.

        Args:
            query: User query string.

        Returns:
            Dict of extracted filter parameters.
        """
        q_lower = query.lower()
        filters = {}

        # Category
        for cat in CATEGORIES:
            if cat in q_lower:
                filters["category"] = cat
                break

        # Sentiment
        if any(w in q_lower for w in ["positive", "optimistic", "upbeat"]):
            filters["sentiment_label"] = "Positive"
        elif any(w in q_lower for w in ["negative", "pessimistic", "bad"]):
            filters["sentiment_label"] = "Negative"
        elif "neutral" in q_lower:
            filters["sentiment_label"] = "Neutral"

        # Keyword extraction (words after "about", "on", "covering")
        keyword_match = re.search(r"(?:about|on|covering|regarding)\s+([a-zA-Z\s]+?)(?:\s+from|\s+in|\s+this|$)", q_lower)
        if keyword_match:
            filters["keyword"] = keyword_match.group(1).strip()

        return filters

    def execute_query(self, query: str) -> dict:
        """
        Parse and execute a natural language query against the corpus.

        Args:
            query: User's natural language question.

        Returns:
            Dict with: intent, filters, n_results, response_text, data.
        """
        if self.df is None:
            return {"error": "No DataFrame loaded. Initialize QueryProcessor with df=your_dataframe."}

        intent  = self.classify_intent(query)
        filters = self.extract_filters(query)
        result  = {"intent": intent, "filters": filters, "query": query}

        # Apply filters
        subset = self.df.copy()
        if "category" in filters:
            subset = subset[subset["category"] == filters["category"]]
        if "sentiment_label" in filters and "sentiment_label" in subset.columns:
            subset = subset[subset["sentiment_label"] == filters["sentiment_label"]]
        if "keyword" in filters and "cleaned_text" in subset.columns:
            subset = subset[subset["cleaned_text"].str.contains(filters["keyword"], case=False, na=False)]

        result["n_results"] = len(subset)

        # Generate response
        if len(subset) == 0:
            result["response_text"] = f"No articles matched your query: '{query}'"
        elif intent == "stats":
            cat_dist  = subset["category"].value_counts().to_dict()
            result["response_text"] = (
                f"Found {len(subset)} articles. Category breakdown: "
                + ", ".join(f"{k}: {v}" for k, v in cat_dist.items())
            )
        elif intent == "filter_by_sentiment":
            label = filters.get("sentiment_label", "all sentiments")
            result["response_text"] = (
                f"Found {len(subset)} {label.lower()} articles"
                + (f" in the '{filters['category']}' category" if "category" in filters else "")
                + f". Average sentiment score: {subset['sentiment_compound'].mean():.3f}"
                if "sentiment_compound" in subset.columns else "."
            )
        else:
            result["response_text"] = (
                f"Found {len(subset)} articles matching your query."
            )

        result["data"] = subset.head(10)[["category", "cleaned_text"]].to_dict("records") \
            if len(subset) > 0 else []

        return result

    def process(self, query: str) -> dict:
        """Alias for execute_query — primary public interface."""
        return self.execute_query(query)
