"""Shared value-application helpers for editing algorithms."""

from __future__ import annotations

import numpy as np


class ApplyValue:
    """Apply edit modes and optional per-vertex weights to numeric arrays."""

    @staticmethod
    def blend(
        existing_values: np.ndarray,
        target_values: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        existing = np.asarray(existing_values, dtype=np.float64)
        target = np.asarray(target_values, dtype=np.float64)
        if target.shape != existing.shape:
            raise ValueError(
                "Target values must match the existing value shape, "
                f"got {target.shape} for {existing.shape}."
            )
        if weights is None:
            return target
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != existing.shape:
            raise ValueError(
                "Apply weights must match the existing value shape, "
                f"got {weight_values.shape} for {existing.shape}."
            )
        return existing + weight_values * (target - existing)

    @staticmethod
    def transform(existing_values: np.ndarray, value: float, mode: str) -> np.ndarray:
        existing = np.asarray(existing_values, dtype=np.float64)
        applied_value = float(value)
        normalized_mode = str(mode)
        if normalized_mode == "Replace":
            return np.full(existing.shape, applied_value, dtype=np.float64)
        if normalized_mode == "Add":
            return existing + applied_value
        if normalized_mode == "Subtract":
            return existing - applied_value
        if normalized_mode == "Minimum":
            return np.minimum(existing, applied_value)
        if normalized_mode == "Maximum":
            return np.maximum(existing, applied_value)
        if normalized_mode == "Multiply":
            return existing * applied_value
        if normalized_mode == "Average":
            return (existing + applied_value) / 2.0
        raise ValueError(f"Unsupported apply mode: {mode!r}")

    @classmethod
    def apply(
        cls,
        existing_values: np.ndarray,
        value: float,
        mode: str,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        existing = np.asarray(existing_values, dtype=np.float64)
        transformed = cls.transform(existing, value, mode)
        return cls.blend(existing, transformed, weights=weights)
