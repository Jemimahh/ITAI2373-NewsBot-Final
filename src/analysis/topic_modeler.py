"""
src/analysis/topic_modeler.py
TopicModeler class — LDA + NMF implementation for NewsBot 2.0.

Wraps sklearn's LDA and NMF with a unified API for training,
topic extraction, document assignment, and visualization.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from config.settings import (
    N_TOPICS, LDA_MAX_ITER, LDA_DOC_PRIOR, LDA_WORD_PRIOR,
    NMF_MAX_ITER, NMF_ALPHA, NMF_L1_RATIO, CLUSTER_K_RANGE, RANDOM_STATE
)

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Unified topic modeling interface supporting LDA and NMF.

    Usage::

        modeler = TopicModeler(n_topics=10, method='both')
        modeler.fit(count_matrix, tfidf_matrix, count_vocab, tfidf_vocab)

        # Get top words
        topics = modeler.get_topic_words(model='lda', n_words=10)

        # Assign documents
        df['lda_topic'] = modeler.get_dominant_topics(model='lda')

        # Cluster by topic distribution
        df['cluster'] = modeler.cluster_documents()

    Attributes:
        n_topics (int): Number of topics to discover.
        method (str): 'lda', 'nmf', or 'both'.
        lda_model: Fitted LDA model (if method includes LDA).
        nmf_model: Fitted NMF model (if method includes NMF).
    """

    def __init__(self, n_topics: int = N_TOPICS, method: str = "both"):
        """
        Initialize TopicModeler.

        Args:
            n_topics: Number of latent topics to discover.
            method: Which model(s) to train — 'lda', 'nmf', or 'both'.
        """
        self.n_topics   = n_topics
        self.method     = method
        self.lda_model  = None
        self.nmf_model  = None
        self.lda_topics_: dict  = {}
        self.nmf_topics_: dict  = {}
        self.lda_doc_topics_: np.ndarray = None
        self.nmf_doc_topics_: np.ndarray = None
        self.count_vocab_: np.ndarray = None
        self.tfidf_vocab_: np.ndarray = None
        self._is_fitted = False

    def fit(
        self,
        count_matrix,
        tfidf_matrix,
        count_vocab: np.ndarray,
        tfidf_vocab: np.ndarray,
    ) -> "TopicModeler":
        """
        Train topic model(s) on document-term matrices.

        Args:
            count_matrix: Sparse count matrix (docs × terms) for LDA.
            tfidf_matrix: Sparse TF-IDF matrix (docs × terms) for NMF.
            count_vocab: Feature names array from CountVectorizer.
            tfidf_vocab: Feature names array from TfidfVectorizer.

        Returns:
            self (for method chaining).
        """
        self.count_vocab_ = count_vocab
        self.tfidf_vocab_ = tfidf_vocab

        if self.method in ("lda", "both"):
            logger.info(f"Training LDA (n_topics={self.n_topics})...")
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=LDA_MAX_ITER,
                learning_method="online",
                learning_offset=50.0,
                doc_topic_prior=LDA_DOC_PRIOR,
                topic_word_prior=LDA_WORD_PRIOR,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            self.lda_doc_topics_ = self.lda_model.fit_transform(count_matrix)
            self.lda_topics_ = self._extract_topics(self.lda_model, count_vocab)
            logger.info(f"LDA perplexity: {self.lda_model.perplexity(count_matrix):.1f}")

        if self.method in ("nmf", "both"):
            logger.info(f"Training NMF (n_topics={self.n_topics})...")
            self.nmf_model = NMF(
                n_components=self.n_topics,
                init="nndsvda",
                solver="cd",
                max_iter=NMF_MAX_ITER,
                alpha_W=NMF_ALPHA,
                alpha_H=NMF_ALPHA,
                l1_ratio=NMF_L1_RATIO,
                random_state=RANDOM_STATE,
            )
            raw = self.nmf_model.fit_transform(tfidf_matrix)
            self.nmf_doc_topics_ = normalize(raw, norm="l1")
            self.nmf_topics_ = self._extract_topics(self.nmf_model, tfidf_vocab)
            logger.info(f"NMF reconstruction error: {self.nmf_model.reconstruction_err_:.1f}")

        self._is_fitted = True
        return self

    def _extract_topics(self, model, vocab: np.ndarray, n_words: int = 15) -> dict:
        """
        Extract top words for each topic from a fitted model.

        Args:
            model: Fitted LDA or NMF model with .components_ attribute.
            vocab: Feature name array.
            n_words: Number of top words per topic.

        Returns:
            Dict mapping topic_id → list of (word, weight) tuples.
        """
        topics = {}
        for i, component in enumerate(model.components_):
            top_idx = component.argsort()[:-n_words - 1:-1]
            topics[i] = [(vocab[j], float(component[j])) for j in top_idx]
        return topics

    def get_topic_words(self, model: str = "nmf", n_words: int = 10) -> dict:
        """
        Return top words for each topic.

        Args:
            model: 'lda' or 'nmf'.
            n_words: Number of words per topic.

        Returns:
            Dict of topic_id → [(word, weight), ...].
        """
        self._check_fitted()
        topics = self.lda_topics_ if model == "lda" else self.nmf_topics_
        return {t: words[:n_words] for t, words in topics.items()}

    def get_dominant_topics(self, model: str = "nmf") -> np.ndarray:
        """
        Return the dominant topic index for each document.

        Args:
            model: 'lda' or 'nmf'.

        Returns:
            Integer array of shape (n_docs,).
        """
        self._check_fitted()
        dist = self.lda_doc_topics_ if model == "lda" else self.nmf_doc_topics_
        return dist.argmax(axis=1)

    def get_topic_confidence(self, model: str = "nmf") -> np.ndarray:
        """
        Return the maximum topic weight (confidence) per document.

        Args:
            model: 'lda' or 'nmf'.

        Returns:
            Float array of shape (n_docs,).
        """
        self._check_fitted()
        dist = self.lda_doc_topics_ if model == "lda" else self.nmf_doc_topics_
        return dist.max(axis=1)

    def auto_label_topics(self, model: str = "nmf") -> dict:
        """
        Auto-generate human-readable labels from top topic words.

        Labels are formed by joining the top 3 non-generic content words.
        Edit the output for presentation purposes.

        Args:
            model: 'lda' or 'nmf'.

        Returns:
            Dict of topic_id → label string.
        """
        generic = {"said", "year", "new", "one", "would", "also", "like", "people", "time"}
        topics  = self.get_topic_words(model, n_words=12)
        labels  = {}
        for t_id, words in topics.items():
            clean = [w for w, _ in words if w not in generic and len(w) > 3]
            labels[t_id] = " / ".join(clean[:3]).title()
        return labels

    def cluster_documents(self, k: int = None) -> np.ndarray:
        """
        Cluster documents using K-Means on NMF topic distributions.

        If k is None, the optimal k is selected via silhouette score
        over the range defined in config.settings.CLUSTER_K_RANGE.

        Args:
            k: Number of clusters, or None for automatic selection.

        Returns:
            Integer cluster label array of shape (n_docs,).
        """
        self._check_fitted(require_nmf=True)
        X = self.nmf_doc_topics_

        if k is None:
            scores = []
            for ki in CLUSTER_K_RANGE:
                km  = KMeans(n_clusters=ki, random_state=RANDOM_STATE, n_init=10)
                lbl = km.fit_predict(X)
                sil = silhouette_score(X, lbl, sample_size=min(1000, len(X)))
                scores.append(sil)
            k = list(CLUSTER_K_RANGE)[int(np.argmax(scores))]
            logger.info(f"Optimal k = {k} (silhouette = {max(scores):.4f})")

        km_final = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        return km_final.fit_predict(X)

    def visualize_topics(
        self,
        model: str = "nmf",
        n_words: int = 10,
        save_path: str = None,
    ) -> None:
        """
        Create interactive topic visualization.

        For LDA: generates a pyLDAvis HTML file (best viewed in browser).
        For NMF: generates a 2×5 bar chart grid of top topic words.

        Args:
            model: 'lda' or 'nmf'.
            n_words: Words per topic bar chart.
            save_path: Optional file path to save the output.
        """
        self._check_fitted()

        if model == "lda":
            try:
                import pyLDAvis
                import pyLDAvis.lda_model as ldavis_lda
                # Requires the original count matrix — pass via fit_transform
                logger.info("pyLDAvis requires count_matrix — call prepare() directly.")
            except ImportError:
                logger.warning("pyLDAvis not installed. Run: pip install pyLDAvis")
            return

        # NMF bar chart grid
        topics  = self.get_topic_words(model, n_words)
        palette = plt.cm.tab10(np.linspace(0, 1, self.n_topics))

        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        axes = axes.flatten()

        for t_id, words in topics.items():
            ax    = axes[t_id]
            terms = [w for w, _ in words][::-1]
            scores = [s for _, s in words][::-1]
            ax.barh(terms, scores, color=palette[t_id], alpha=0.85)
            ax.set_title(f"NMF Topic {t_id}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Weight", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", alpha=0.3)

        plt.suptitle(f"{model.upper()} Topics — NewsBot 2.0", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved: {save_path}")
        plt.show()

    def get_evaluation_metrics(self, count_matrix) -> dict:
        """
        Return model evaluation metrics.

        Args:
            count_matrix: The count matrix used to fit LDA.

        Returns:
            Dict with perplexity (LDA), reconstruction_error (NMF).
        """
        metrics = {}
        if self.lda_model:
            metrics["lda_perplexity"]       = self.lda_model.perplexity(count_matrix)
            metrics["lda_log_likelihood"]   = self.lda_model.score(count_matrix)
        if self.nmf_model:
            metrics["nmf_reconstruction_error"] = self.nmf_model.reconstruction_err_
        return metrics

    def _check_fitted(self, require_nmf: bool = False) -> None:
        if not self._is_fitted:
            raise RuntimeError("TopicModeler must be fitted before calling this method. Run .fit() first.")
        if require_nmf and self.nmf_doc_topics_ is None:
            raise RuntimeError("NMF model required. Initialize with method='nmf' or 'both'.")
