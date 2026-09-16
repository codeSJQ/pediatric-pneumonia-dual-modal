"""Smote-Voting-Stacking classifier used by the main experiment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from mlxtend.classifier import StackingCVClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


MODEL_ORDER = ("LR", "RF", "KNN", "SVM", "XGB")


def base_classifiers() -> Dict[str, BaseEstimator]:
    """Return fresh base classifiers with the manuscript hyperparameters."""
    return {
        "LR": LogisticRegression(
            C=1.06, max_iter=2952, solver="liblinear", random_state=1412
        ),
        "RF": RandomForestClassifier(
            n_estimators=167,
            max_features=0.46,
            max_depth=7,
            random_state=3,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=1, n_jobs=1),
        "SVM": SVC(C=417022, probability=True, random_state=42),
        "XGB": XGBClassifier(
            colsample_bytree=0.585,
            learning_rate=0.037,
            max_depth=7,
            subsample=0.728,
            verbosity=0,
            random_state=42,
        ),
    }


def build_svs_classifier(
    resampled_features: np.ndarray,
    resampled_labels: np.ndarray,
    stacking_seed: int = 52,
) -> Tuple[StackingCVClassifier, Dict[str, float]]:
    """Build the weighted voting-stacking model and return its learned weights."""
    definitions = base_classifiers()
    training_accuracy = {}
    for name in MODEL_ORDER:
        classifier = clone(definitions[name])
        classifier.fit(resampled_features, resampled_labels)
        training_accuracy[name] = min(
            classifier.score(resampled_features, resampled_labels), 0.9999
        )

    raw_weights = {
        name: score / (1.0 - score) for name, score in training_accuracy.items()
    }
    total_weight = sum(raw_weights.values())
    weights = {name: raw_weights[name] / total_weight for name in MODEL_ORDER}
    ranked_names = sorted(MODEL_ORDER, key=weights.get, reverse=True)

    def fresh(name: str) -> BaseEstimator:
        return clone(definitions[name])

    alpha = VotingClassifier(
        [(ranked_names[0], fresh(ranked_names[0]))],
        voting="soft",
        weights=[weights[ranked_names[0]]],
    )
    beta = VotingClassifier(
        [(name, fresh(name)) for name in ranked_names[1:4]],
        voting="soft",
        weights=[weights[name] for name in ranked_names[1:4]],
    )
    gamma = VotingClassifier(
        [(ranked_names[4], fresh(ranked_names[4]))],
        voting="soft",
        weights=[weights[ranked_names[4]]],
    )
    delta = VotingClassifier(
        [(name, fresh(name)) for name in ranked_names], voting="soft"
    )
    classifier = StackingCVClassifier(
        [alpha, beta, gamma],
        meta_classifier=delta,
        cv=5,
        use_probas=True,
        random_state=stacking_seed,
    )
    return classifier, weights


def build_explanation_forest() -> RandomForestClassifier:
    """Return the fixed Random Forest used for modality-level OOF SHAP."""
    return clone(base_classifiers()["RF"])
