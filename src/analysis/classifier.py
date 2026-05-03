"""
src/analysis/classifier.py
Multi-class news article classifier with confidence scoring.
Uses TF-IDF features + Logistic Regression / SVM ensemble.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from config.settings import CATEGORIES, RANDOM_STATE

logger = logging.getLogger(__name__)


class NewsClassifier:
    """
    Multi-class news article classifier.

    Supports Logistic Regression and Linear SVC with confidence scoring
    via Platt scaling calibration.

    Usage::

        clf = NewsClassifier(model_type='logreg')
        clf.fit(X_train, y_train)
        labels, confidences = clf.predict_with_confidence(X_test)
        print(clf.evaluate(X_test, y_test))
    """

    SUPPORTED_MODELS = ("logreg", "svm")

    def __init__(self, model_type: str = "logreg"):
        """
        Args:
            model_type: 'logreg' for Logistic Regression, 'svm' for Linear SVC.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.model      = None
        self._is_fitted = False

    def _build_model(self):
        if self.model_type == "logreg":
            return LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                multi_class="multinomial", random_state=RANDOM_STATE
            )
        # SVM: calibrate for probability estimates
        svc = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
        return CalibratedClassifierCV(svc, cv=3)

    def fit(self, X, y) -> "NewsClassifier":
        """
        Train the classifier.

        Args:
            X: Feature matrix (TF-IDF sparse or dense).
            y: Label array.

        Returns:
            self.
        """
        self.model = self._build_model()
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(f"NewsClassifier ({self.model_type}) fitted.")
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict category labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted label strings.
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probability estimates.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_docs, n_classes).
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and return confidence scores.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (predicted_labels, confidence_scores).
        """
        self._check_fitted()
        proba  = self.predict_proba(X)
        labels = self.model.classes_[proba.argmax(axis=1)]
        confs  = proba.max(axis=1)
        return labels, confs

    def evaluate(self, X, y_true) -> dict:
        """
        Generate classification report and confusion matrix.

        Args:
            X: Feature matrix.
            y_true: Ground-truth labels.

        Returns:
            Dict with keys: report (str), confusion_matrix (ndarray).
        """
        self._check_fitted()
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred, target_names=CATEGORIES)
        cm     = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
        print(report)
        return {"report": report, "confusion_matrix": cm}

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Run stratified k-fold cross-validation.

        Args:
            X: Full feature matrix.
            y: Full label array.
            cv: Number of folds.

        Returns:
            Dict with mean and std accuracy.
        """
        model  = self._build_model()
        skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        result = {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}
        logger.info(f"CV accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        return result

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call .fit() first.")
