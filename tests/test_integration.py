"""
tests/test_integration.py
Integration tests — validates that modules chain correctly.
Run: pytest tests/test_integration.py -v
"""

import pytest
import pandas as pd
from src.data_processing.text_preprocessor import clean_text, extract_named_entities
from src.analysis.sentiment_analyzer import analyze_sentiment
from src.data_processing.data_validator import validate_raw_dataframe, DataValidationError
from src.conversation.query_processor import QueryProcessor

SAMPLE_DF = pd.DataFrame({
    "text": [
        "The Federal Reserve raised interest rates as inflation persisted.",
        "Apple unveiled its new AI-powered chip at the developer conference.",
        "Parliament passed the education reform bill after heated debate.",
        "Manchester United won the Premier League championship in extra time.",
        "Oscar-winning director announced a new film starring major Hollywood stars.",
    ],
    "category": ["business", "tech", "politics", "sport", "entertainment"]
})


class TestPreprocessingToSentimentPipeline:
    def test_clean_then_sentiment(self):
        """Cleaning output should work as sentiment input."""
        for text in SAMPLE_DF["text"]:
            cleaned   = clean_text(text)
            sentiment = analyze_sentiment(cleaned)
            assert "sentiment_label" in sentiment
            assert sentiment["sentiment_label"] in ("Positive", "Neutral", "Negative")
            assert -1.0 <= sentiment["compound"] <= 1.0

    def test_entities_from_raw_text(self):
        """NER should return dicts from raw text."""
        for text in SAMPLE_DF["text"]:
            ents = extract_named_entities(text)
            assert isinstance(ents, dict)


class TestDataValidation:
    def test_valid_dataframe_passes(self):
        report = validate_raw_dataframe(SAMPLE_DF)
        assert report["stats"]["n_rows"] == 5

    def test_missing_column_raises(self):
        bad_df = SAMPLE_DF.drop(columns=["category"])
        with pytest.raises(DataValidationError):
            validate_raw_dataframe(bad_df)


class TestQueryProcessor:
    def test_intent_classification(self):
        qp = QueryProcessor(SAMPLE_DF)
        assert qp.classify_intent("show me positive tech news") in (
            "filter_by_sentiment", "filter_by_category"
        )

    def test_filter_extraction_category(self):
        qp = QueryProcessor()
        filters = qp.extract_filters("show me tech articles")
        assert filters.get("category") == "tech"

    def test_filter_extraction_sentiment(self):
        qp = QueryProcessor()
        filters = qp.extract_filters("find positive business news")
        assert filters.get("sentiment_label") == "Positive"

    def test_execute_query_with_df(self):
        qp = QueryProcessor(SAMPLE_DF)
        result = qp.process("tech articles")
        assert "n_results" in result
        assert result["n_results"] >= 0
