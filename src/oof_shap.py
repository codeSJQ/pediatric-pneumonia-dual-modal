"""Compute modality-level SHAP values on held-out samples from each fixed fold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE

from evaluation import load_config, load_dataset, resolve_path
from svs_classifier import build_explanation_forest

def positive_class_values(
    raw_values: object, sample_count: int, feature_count: int
) -> np.ndarray:
    """Normalize SHAP version-specific output to samples by features for class 1."""
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            if values.shape[:2] == (sample_count, feature_count):
                values = values[:, :, 1]
            elif values.shape[1:] == (sample_count, feature_count):
                values = values[1]
    if values.shape != (sample_count, feature_count):
        raise ValueError(
            f"Unexpected SHAP array shape {values.shape}; expected "
            f"({sample_count}, {feature_count})."
        )
    return values.astype(float, copy=False)


def explain_modality(
    name: str,
    features: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    folds: np.ndarray,
    sample_ids: np.ndarray,
    output_directory: Path,
    fold_count: int,
    smote_seed: int,
    top_n: int,
) -> None:
    sample_count, feature_count = features.shape
    oof_values = np.full((sample_count, feature_count), np.nan)
    occurrences = np.zeros(sample_count, dtype=int)

    for fold in range(1, fold_count + 1):
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        resampled_features, resampled_labels = SMOTE(
            random_state=smote_seed
        ).fit_resample(features[train_indices], labels[train_indices])
        forest = build_explanation_forest()
        forest.fit(resampled_features, resampled_labels)
        raw_values = shap.TreeExplainer(forest).shap_values(features[test_indices])
        oof_values[test_indices] = positive_class_values(
            raw_values, len(test_indices), feature_count
        )
        occurrences[test_indices] += 1
        print(f"{name} fold {fold}: explained {len(test_indices)} held-out samples")

    if not np.all(occurrences == 1) or not np.isfinite(oof_values).all():
        raise AssertionError("Every sample must receive one finite OOF SHAP vector.")

    mean_absolute = np.mean(np.abs(oof_values), axis=0)
    ranking = np.argsort(mean_absolute)[::-1]
    importance = pd.DataFrame(
        {
            "rank": np.arange(1, feature_count + 1),
            "feature": np.asarray(feature_names)[ranking],
            "mean_absolute_oof_shap": mean_absolute[ranking],
        }
    )
    importance.to_csv(output_directory / f"shap_{name.lower()}_importance.csv", index=False)
    np.savez_compressed(
        output_directory / f"shap_{name.lower()}_oof_values.npz",
        shap_values=oof_values,
        feature_values=features,
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        feature_names=np.asarray(feature_names),
    )

    selected = ranking[: min(top_n, feature_count)]
    shap.summary_plot(
        oof_values[:, selected],
        features[:, selected],
        feature_names=np.asarray(feature_names)[selected],
        max_display=len(selected),
        show=False,
    )
    plt.gcf().set_size_inches(10, 8)
    plt.title(f"{name} feature importance (out-of-fold SHAP)")
    plt.tight_layout()
    plt.savefig(output_directory / f"shap_{name.lower()}_top{len(selected)}.png", dpi=300)
    plt.close()


def run(config_path: Path) -> None:
    config = load_config(config_path)
    data = load_dataset(config, config_path)
    output_directory = resolve_path(config_path, config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    arguments = {
        "labels": data["labels"],
        "folds": data["folds"],
        "sample_ids": data["sample_ids"],
        "output_directory": output_directory,
        "fold_count": int(config["cross_validation"]["folds"]),
        "smote_seed": int(config["training"]["smote_seed"]),
        "top_n": int(config["explanation"]["top_features"]),
    }
    explain_modality(
        "ASM", data["asm"], data["asm_feature_names"], **arguments
    )
    explain_modality(
        "FOSM", data["fosm"], data["fosm_feature_names"], **arguments
    )
    print(f"SHAP outputs written to {output_directory}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments().config)
