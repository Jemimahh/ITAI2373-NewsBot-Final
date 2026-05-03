"""
tests/test_topic_modeling.py
Unit tests for TopicModeler class.
Run: pytest tests/test_topic_modeling.py -v
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from src.analysis.topic_modeler import TopicModeler

SAMPLE_TEXTS = [
    "The stock market fell today as investors reacted to inflation data",
    "Technology companies reported strong earnings driven by AI demand",
    "The prime minister announced new economic policy reforms",
    "Sports teams competed in the championship finals",
    "Film studios released blockbuster movies this summer",
    "Central banks raised interest rates to combat inflation",
    "Silicon Valley startups raised millions in venture capital funding",
    "Parliament debated the new immigration legislation",
    "The football league announced its new season schedule",
    "Streaming platforms released original series and documentaries",
]


@pytest.fixture
def fitted_modeler():
    """Return a fitted TopicModeler with n_topics=3 for speed."""
    count_vec  = CountVectorizer(min_df=1, stop_words="english")
    tfidf_vec  = TfidfVectorizer(min_df=1, stop_words="english")
    cm  = count_vec.fit_transform(SAMPLE_TEXTS)
    tm  = tfidf_vec.fit_transform(SAMPLE_TEXTS)
    cv  = count_vec.get_feature_names_out()
    tv  = tfidf_vec.get_feature_names_out()

    modeler = TopicModeler(n_topics=3, method="both")
    modeler.fit(cm, tm, cv, tv)
    return modeler


class TestTopicModelerFit:
    def test_fit_returns_self(self, fitted_modeler):
        assert fitted_modeler._is_fitted

    def test_lda_model_exists(self, fitted_modeler):
        assert fitted_modeler.lda_model is not None

    def test_nmf_model_exists(self, fitted_modeler):
        assert fitted_modeler.nmf_model is not None

    def test_doc_topic_shape_lda(self, fitted_modeler):
        assert fitted_modeler.lda_doc_topics_.shape == (len(SAMPLE_TEXTS), 3)

    def test_doc_topic_shape_nmf(self, fitted_modeler):
        assert fitted_modeler.nmf_doc_topics_.shape == (len(SAMPLE_TEXTS), 3)


class TestTopicModelerTopicWords:
    def test_get_topic_words_lda(self, fitted_modeler):
        topics = fitted_modeler.get_topic_words(model="lda", n_words=5)
        assert len(topics) == 3
        for t_id, words in topics.items():
            assert len(words) <= 5
            assert all(isinstance(w, str) and isinstance(s, float) for w, s in words)

    def test_get_topic_words_nmf(self, fitted_modeler):
        topics = fitted_modeler.get_topic_words(model="nmf", n_words=5)
        assert len(topics) == 3


class TestTopicModelerDominantTopics:
    def test_dominant_topics_length(self, fitted_modeler):
        topics = fitted_modeler.get_dominant_topics(model="nmf")
        assert len(topics) == len(SAMPLE_TEXTS)

    def test_dominant_topics_range(self, fitted_modeler):
        topics = fitted_modeler.get_dominant_topics(model="lda")
        assert all(0 <= t < 3 for t in topics)


class TestTopicModelerClustering:
    def test_cluster_returns_labels(self, fitted_modeler):
        labels = fitted_modeler.cluster_documents(k=2)
        assert len(labels) == len(SAMPLE_TEXTS)
        assert set(labels).issubset({0, 1})


class TestTopicModelerNotFitted:
    def test_raises_if_not_fitted(self):
        modeler = TopicModeler()
        with pytest.raises(RuntimeError):
            modeler.get_topic_words()
