"""
Classifier Architecture Module
Implements classical machine learning classifiers trained strictly on DIP feature vectors.
Supports Random Forest, Support Vector Machine (RBF/Linear), and K-Nearest Neighbors.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


class SignLanguageClassifier:
    """
    Classical Machine Learning classifier wrapper for DIP sign language recognition.
    """

    def __init__(self, model_type="rf", **kwargs):
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        self.classes_ = []
        self.model = self._create_model(self.model_type, **kwargs)
        self.is_trained = False
        self.metadata = {}

    def _create_model(self, model_type, **kwargs):
        if model_type in ("rf", "random_forest"):
            n_estimators = kwargs.get("n_estimators", 150)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=kwargs.get("max_depth", None),
                random_state=42,
                n_jobs=-1
            )
        elif model_type in ("svm", "svc"):
            return SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 10.0),
                probability=True,
                random_state=42
            )
        elif model_type in ("knn", "k_nearest_neighbors"):
            return KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 3),
                weights=kwargs.get("weights", "distance")
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        """
        Fits the scaler and classical classifier on extracted DIP feature vectors.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True

        self.metadata = {
            "model_type": self.model_type,
            "classes": self.classes_,
            "num_classes": len(self.classes_),
            "feature_dim": int(X.shape[1]),
            "num_samples": int(X.shape[0])
        }
        return self

    def predict(self, X):
        """
        Predicts sign class labels for feature vector(s).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Returns class probabilities for feature vector(s).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X_scaled)
            probs = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
            for i, p in enumerate(preds):
                idx = self.classes_.index(p)
                probs[i, idx] = 1.0
            return probs

    def evaluate(self, X_test, y_test):
        """
        Computes accuracy, precision, recall, f1-score, and confusion matrix.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        return {
            "accuracy": float(round(acc, 4)),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1_score": float(round(f1, 4)),
            "classification_report": report,
            "confusion_matrix": cm,
            "classes": self.classes_
        }

    def save(self, file_path):
        """
        Serializes trained model, scaler, and metadata to disk.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "classes_": self.classes_,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "metadata": self.metadata
        }, path)

    @classmethod
    def load(cls, file_path):
        """
        Loads saved model from disk.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        data = joblib.load(path)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.classes_ = data["classes_"]
        instance.is_trained = data["is_trained"]
        instance.metadata = data.get("metadata", {})
        return instance
