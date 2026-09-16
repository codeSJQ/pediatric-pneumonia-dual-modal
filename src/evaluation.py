"""Run fixed-fold out-of-fold training for the final CA-LMF + SVS model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_fusion import CALMFFusion
from svs_classifier import build_svs_classifier


def load_config(path: Path) -> dict:
    """Load the JSON-compatible YAML configuration without extra dependencies."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_path(config_path: Path, configured_path: str) -> Path:
    repository_root = config_path.resolve().parent.parent
    return repository_root / configured_path


def load_dataset(config: Mapping, config_path: Path) -> Dict[str, object]:
    """Load, align, and validate the anonymized matrices and fixed folds."""
    data_config = config["data"]
    id_column = data_config["id_column"]
    label_column = data_config["label_column"]

    asm_frame = pd.read_csv(resolve_path(config_path, data_config["asm_features"]))
    fosm_frame = pd.read_csv(resolve_path(config_path, data_config["fosm_features"]))
    labels_frame = pd.read_csv(resolve_path(config_path, data_config["labels"]))
    folds_frame = pd.read_csv(resolve_path(config_path, data_config["fold_indices"]))

    frames = {
        "ASM features": asm_frame,
        "FOSM features": fosm_frame,
        "labels": labels_frame,
        "fold indices": folds_frame,
    }
    for name, frame in frames.items():
        if id_column not in frame.columns:
            raise ValueError(f"{name} is missing the {id_column!r} column.")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate sample identifiers.")

    sample_ids = labels_frame[id_column].astype(str)
    expected_ids = set(sample_ids)
    for name, frame in frames.items():
        if set(frame[id_column].astype(str)) != expected_ids:
            raise ValueError(f"{name} does not contain the same sample identifiers.")

    def align(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{id_column: frame[id_column].astype(str)}).set_index(
            id_column
        ).loc[sample_ids]

    asm_aligned = align(asm_frame)
    fosm_aligned = align(fosm_frame)
    labels_aligned = align(labels_frame)
    folds_aligned = align(folds_frame)

    asm = asm_aligned.to_numpy(dtype=float)
    fosm = fosm_aligned.to_numpy(dtype=float)
    labels = labels_aligned[label_column].to_numpy(dtype=int)
    folds = folds_aligned["fold"].to_numpy(dtype=int)

    if not np.isfinite(asm).all() or not np.isfinite(fosm).all():
        raise ValueError("Feature matrices must contain only finite numeric values.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must contain both binary classes 0 and 1.")
    expected_folds = set(range(1, int(config["cross_validation"]["folds"]) + 1))
    if set(np.unique(folds)) != expected_folds:
        raise ValueError(f"Fold indices must contain exactly {sorted(expected_folds)}.")

    return {
        "sample_ids": sample_ids.to_numpy(),
        "asm": asm,
        "fosm": fosm,
        "labels": labels,
        "folds": folds,
        "asm_feature_names": asm_aligned.columns.to_list(),
        "fosm_feature_names": fosm_aligned.columns.to_list(),
    }


def select_ranges(matrix: np.ndarray, ranges: Sequence[Sequence[int]]) -> np.ndarray:
    """Select and concatenate zero-based, stop-exclusive column ranges."""
    selected = [matrix[:, int(start) : int(stop)] for start, stop in ranges]
    if not selected or any(part.shape[1] == 0 for part in selected):
        raise ValueError("Each configured feature range must select at least one column.")
    return np.hstack(selected)


def metric_values(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "sensitivity": recall_score(labels, predictions),
        "specificity": recall_score(labels, predictions, pos_label=0),
        "f1": f1_score(labels, predictions),
        "auc": roc_auc_score(labels, probabilities),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
    }


def summarize_metrics(fold_metrics: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(fold_metrics)
    return pd.DataFrame(
        {
            "metric": frame.columns,
            "mean": [frame[column].mean() for column in frame.columns],
            "std": [frame[column].std(ddof=0) for column in frame.columns],
        }
    )


def run_evaluation(config_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    data = load_dataset(config, config_path)
    sample_ids = data["sample_ids"]
    asm = data["asm"]
    fosm = data["fosm"]
    labels = data["labels"]
    folds = data["folds"]

    fusion_config = config["fusion"]
    asm_frequency = select_ranges(asm, fusion_config["asm_frequency_ranges"])
    fosm_frequency = select_ranges(fosm, fusion_config["fosm_frequency_ranges"])

    prediction_rows: List[dict] = []
    metric_rows: List[dict] = []
    expected_folds = range(1, int(config["cross_validation"]["folds"]) + 1)
    for fold in expected_folds:
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        fusion = CALMFFusion(
            latent_dim=int(fusion_config["latent_dim"]),
            regularization=float(fusion_config["regularization"]),
        )
        train_matrices = (
            asm[train_indices],
            fosm[train_indices],
            asm_frequency[train_indices],
            fosm_frequency[train_indices],
        )
        test_matrices = (
            asm[test_indices],
            fosm[test_indices],
            asm_frequency[test_indices],
            fosm_frequency[test_indices],
        )
        fused_train, fused_test = fusion.fit_transform_pair(
            train_matrices, test_matrices
        )
        resampled_features, resampled_labels = SMOTE(
            random_state=int(config["training"]["smote_seed"])
        ).fit_resample(fused_train, labels[train_indices])
        classifier, weights = build_svs_classifier(
            resampled_features,
            resampled_labels,
            stacking_seed=int(config["training"]["stacking_seed"]),
        )
        classifier.fit(resampled_features, resampled_labels)
        predictions = classifier.predict(fused_test).astype(int)
        probabilities = classifier.predict_proba(fused_test)[:, 1]

        fold_metric = {"fold": fold}
        fold_metric.update(
            metric_values(labels[test_indices], predictions, probabilities)
        )
        metric_rows.append(fold_metric)
        for position, sample_index in enumerate(test_indices):
            prediction_rows.append(
                {
                    "sample_id": sample_ids[sample_index],
                    "fold": fold,
                    "y_true": int(labels[sample_index]),
                    "y_pred": int(predictions[position]),
                    "y_probability": float(probabilities[position]),
                }
            )
        weight_text = ", ".join(f"{name}={value:.4f}" for name, value in weights.items())
        print(f"Fold {fold}: train={len(train_indices)}, test={len(test_indices)}; {weight_text}")

    predictions_frame = pd.DataFrame(prediction_rows).sort_values("sample_id")
    if len(predictions_frame) != len(sample_ids):
        raise AssertionError("OOF prediction count does not match the dataset.")
    if predictions_frame["sample_id"].nunique() != len(sample_ids):
        raise AssertionError("Each sample must have exactly one OOF prediction.")

    fold_metrics_frame = pd.DataFrame(metric_rows)
    summary_frame = summarize_metrics(
        fold_metrics_frame.drop(columns="fold").to_dict(orient="records")
    )
    pooled = metric_values(
        predictions_frame["y_true"].to_numpy(),
        predictions_frame["y_pred"].to_numpy(),
        predictions_frame["y_probability"].to_numpy(),
    )
    summary_frame["pooled_oof"] = summary_frame["metric"].map(pooled)

    output_directory = resolve_path(config_path, config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(output_directory / "oof_predictions.csv", index=False)
    fold_metrics_frame.to_csv(output_directory / "fold_metrics.csv", index=False)
    summary_frame.to_csv(output_directory / "summary_metrics.csv", index=False)
    print("\nFive-fold summary")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nOutputs written to {output_directory}")
    return predictions_frame, fold_metrics_frame, summary_frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml",
        help="Path to the JSON-compatible YAML experiment configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_arguments().config)
