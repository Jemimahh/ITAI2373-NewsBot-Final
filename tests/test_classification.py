"""
tests/test_classification.py
Unit tests for NewsClassifier.
Run: pytest tests/test_classification.py -v
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.analysis.classifier import NewsClassifier

TEXTS = [
    "The stock market rose sharply on strong earnings reports",
    "AI companies are raising billions in venture capital",
    "Parliament voted on the new healthcare bill",
    "The football championship attracted millions of viewers",
    "Hollywood released its biggest blockbuster of the summer",
]
LABELS = ["business", "tech", "politics", "sport", "entertainment"]


@pytest.fixture
def fitted_clf():
    vec  = TfidfVectorizer(min_df=1)
    X    = vec.fit_transform(TEXTS)
    clf  = NewsClassifier(model_type="logreg")
    clf.fit(X, LABELS)
    return clf, X


class TestNewsClassifierInit:
    def test_valid_model_type(self):
        NewsClassifier("logreg")
        NewsClassifier("svm")

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            NewsClassifier("invalid")


class TestNewsClassifierFit:
    def test_fit_sets_fitted_flag(self, fitted_clf):
        clf, _ = fitted_clf
        assert clf._is_fitted

    def test_predict_returns_array(self, fitted_clf):
        clf, X = fitted_clf
        preds = clf.predict(X)
        assert len(preds) == len(TEXTS)

    def test_predict_proba_shape(self, fitted_clf):
        clf, X = fitted_clf
        proba = clf.predict_proba(X)
        assert proba.shape[0] == len(TEXTS)

    def test_confidence_in_range(self, fitted_clf):
        clf, X = fitted_clf
        _, confs = clf.predict_with_confidence(X)
        assert all(0.0 <= c <= 1.0 for c in confs)


class TestNewsClassifierNotFitted:
    def test_predict_raises(self):
        clf = NewsClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((3, 10)))
