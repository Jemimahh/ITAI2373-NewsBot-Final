"""
tests/test_preprocessing.py
Unit tests for text preprocessing functions.
Run: pytest tests/test_preprocessing.py -v
"""

import pytest
from src.data_processing.text_preprocessor import (
    clean_text,
    expand_contractions,
    tokenize_and_process,
    extract_named_entities,
)


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_removes_url(self):
        result = clean_text("Visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_removes_html(self):
        result = clean_text("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result
        assert "hello" in result

    def test_removes_special_chars(self):
        result = clean_text("Price: $50.99 — great deal!")
        assert "$" not in result
        assert "—" not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        result = clean_text(None)
        assert isinstance(result, str)

    def test_expands_contractions(self):
        result = clean_text("don't can't won't")
        assert "do not" in result or "cannot" in result


class TestExpandContractions:
    def test_basic_contraction(self):
        result = expand_contractions("I'm going to the store")
        assert "I'm" not in result

    def test_no_contraction(self):
        text = "The company reported earnings"
        assert expand_contractions(text) == text


class TestTokenizeAndProcess:
    def test_returns_list(self):
        result = tokenize_and_process("The economy grew last year")
        assert isinstance(result, list)

    def test_removes_stopwords(self):
        result = tokenize_and_process("the and or but", remove_stopwords=True)
        assert len(result) == 0

    def test_lemmatizes(self):
        result = tokenize_and_process("running runners ran", lemmatize=True)
        assert "run" in result or "runner" in result

    def test_minimum_length_filter(self):
        result = tokenize_and_process("a bb ccc dddd")
        for token in result:
            assert len(token) >= 2


class TestExtractNamedEntities:
    def test_returns_dict(self):
        result = extract_named_entities("Apple announced new products in California")
        assert isinstance(result, dict)

    def test_finds_org(self):
        result = extract_named_entities("Apple Inc. is headquartered in Cupertino")
        # spaCy should catch Apple or Apple Inc.
        all_ents = [e for lst in result.values() for e in lst]
        assert any("apple" in e.lower() for e in all_ents)

    def test_handles_empty(self):
        result = extract_named_entities("")
        assert isinstance(result, dict)
