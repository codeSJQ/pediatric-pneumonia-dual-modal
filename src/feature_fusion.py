"""Leakage-free cross-attention and low-rank multimodal feature fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ArrayPair = Tuple[np.ndarray, np.ndarray]


@dataclass
class CALMFFusion:
    """Fit the CA-LMF transformation on one outer training fold.

    The implementation follows the experiment pipeline used in the manuscript:
    full ASM and FOSM features are retained, a biased element-wise low-rank
    interaction is appended, and regularized CCA interactions between the two
    frequency-domain subsets form the cross-attention features.
    """

    latent_dim: int = 20
    regularization: float = 1e-2

    def fit(
        self,
        asm: np.ndarray,
        fosm: np.ndarray,
        asm_frequency: np.ndarray,
        fosm_frequency: np.ndarray,
    ) -> "CALMFFusion":
        arrays = (asm, fosm, asm_frequency, fosm_frequency)
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError("All training matrices must contain the same samples.")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive.")

        self._asm_scaler, self._asm_pca = self._fit_pca(asm, self.latent_dim)
        self._fosm_scaler, self._fosm_pca = self._fit_pca(fosm, self.latent_dim)
        self._ca_asm_scaler, self._ca_asm_pca = self._fit_pca(
            asm_frequency, self.latent_dim
        )
        self._ca_fosm_scaler, self._ca_fosm_pca = self._fit_pca(
            fosm_frequency, self.latent_dim
        )

        asm_latent = self._project(
            asm_frequency, self._ca_asm_scaler, self._ca_asm_pca
        )
        fosm_latent = self._project(
            fosm_frequency, self._ca_fosm_scaler, self._ca_fosm_pca
        )
        shared_dim = min(asm_latent.shape[1], fosm_latent.shape[1])
        asm_latent = asm_latent[:, :shared_dim]
        fosm_latent = fosm_latent[:, :shared_dim]

        self._ca_asm_mean = asm_latent.mean(axis=0)
        self._ca_fosm_mean = fosm_latent.mean(axis=0)
        asm_centered = asm_latent - self._ca_asm_mean
        fosm_centered = fosm_latent - self._ca_fosm_mean
        sample_count = asm_centered.shape[0]

        asm_covariance = (
            asm_centered.T @ asm_centered / sample_count
            + self.regularization * np.eye(shared_dim)
        )
        fosm_covariance = (
            fosm_centered.T @ fosm_centered / sample_count
            + self.regularization * np.eye(shared_dim)
        )
        cross_covariance = asm_centered.T @ fosm_centered / sample_count

        asm_inverse_root = self._inverse_square_root(asm_covariance)
        fosm_inverse_root = self._inverse_square_root(fosm_covariance)
        left, _, right_transposed = np.linalg.svd(
            asm_inverse_root @ cross_covariance @ fosm_inverse_root,
            full_matrices=False,
        )
        self._query_projection = asm_inverse_root @ left
        self._key_projection = fosm_inverse_root @ right_transposed.T
        self._upper_triangle = np.triu_indices(shared_dim)
        return self

    def transform(
        self,
        asm: np.ndarray,
        fosm: np.ndarray,
        asm_frequency: np.ndarray,
        fosm_frequency: np.ndarray,
    ) -> np.ndarray:
        """Transform samples using parameters learned only from training data."""
        self._check_is_fitted()
        if len({x.shape[0] for x in (asm, fosm, asm_frequency, fosm_frequency)}) != 1:
            raise ValueError("All matrices must contain the same samples.")

        asm_low_rank = self._project(asm, self._asm_scaler, self._asm_pca)
        fosm_low_rank = self._project(fosm, self._fosm_scaler, self._fosm_pca)
        lmf_interaction = self._biased_product((asm_low_rank, fosm_low_rank))

        asm_ca = self._project(
            asm_frequency, self._ca_asm_scaler, self._ca_asm_pca
        )[:, : len(self._ca_asm_mean)]
        fosm_ca = self._project(
            fosm_frequency, self._ca_fosm_scaler, self._ca_fosm_pca
        )[:, : len(self._ca_fosm_mean)]
        queries = (asm_ca - self._ca_asm_mean) @ self._query_projection
        keys = (fosm_ca - self._ca_fosm_mean) @ self._key_projection
        row_indices, column_indices = self._upper_triangle
        ca_interaction = queries[:, row_indices] * keys[:, column_indices]

        return np.hstack((asm, fosm, lmf_interaction, ca_interaction))

    def fit_transform_pair(
        self,
        train: Sequence[np.ndarray],
        test: Sequence[np.ndarray],
    ) -> ArrayPair:
        """Fit on four training matrices and transform train and test matrices."""
        if len(train) != 4 or len(test) != 4:
            raise ValueError("train and test must each contain four matrices.")
        self.fit(*train)
        return self.transform(*train), self.transform(*test)

    @staticmethod
    def _fit_pca(data: np.ndarray, requested_dim: int) -> Tuple[StandardScaler, PCA]:
        component_count = min(requested_dim, data.shape[0], data.shape[1])
        scaler = StandardScaler().fit(data)
        pca = PCA(n_components=component_count, random_state=42).fit(
            scaler.transform(data)
        )
        return scaler, pca

    @staticmethod
    def _project(data: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
        return pca.transform(scaler.transform(data))

    @staticmethod
    def _biased_product(modalities: Sequence[np.ndarray]) -> np.ndarray:
        width = modalities[0].shape[1] + 1
        interaction = np.ones((modalities[0].shape[0], width), dtype=float)
        for modality in modalities:
            if modality.shape[1] + 1 != width:
                raise ValueError("Low-rank modality representations must have equal width.")
            interaction *= np.hstack((modality, np.ones((len(modality), 1))))
        return interaction

    @staticmethod
    def _inverse_square_root(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        stable_values = np.maximum(eigenvalues, 1e-10)
        return eigenvectors @ np.diag(1.0 / np.sqrt(stable_values)) @ eigenvectors.T

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_query_projection"):
            raise RuntimeError("CALMFFusion.fit must be called before transform.")
