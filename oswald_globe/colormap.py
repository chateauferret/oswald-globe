"""Colormap utilities for globe visualization."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union

import numpy as np
from PySide6.QtCore import QFile, QIODevice

try:
    from . import resources_rc  # noqa: F401
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe import resources_rc  # noqa: F401

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:  # pragma: no cover - exercised in matplotlib-free environments
    plt = None
    LinearSegmentedColormap = Any  # type: ignore[assignment]

LEGENDS_DIR = Path(__file__).resolve().parent / "resources" / "legends"
DEFAULT_TOPO_LEGEND = ":/legends/topography.txt"


class _FallbackColormap:
    def __init__(self, name: str, anchors: np.ndarray):
        self.name = name
        self._anchors = np.ascontiguousarray(anchors, dtype=np.float32)

    def __call__(self, values):
        arr = np.asarray(values, dtype=np.float32)
        flat = np.clip(arr.reshape(-1), 0.0, 1.0)
        positions = self._anchors[:, 0]
        rgba = np.empty((flat.size, 4), dtype=np.float32)
        for idx in range(3):
            rgba[:, idx] = np.interp(flat, positions, self._anchors[:, idx + 1])
        rgba[:, 3] = 1.0
        return rgba.reshape(arr.shape + (4,))


def _terrain_anchors() -> np.ndarray:
    return np.array(
        [
            [0.00, 0.00, 0.20, 0.55],
            [0.20, 0.02, 0.35, 0.70],
            [0.40, 0.12, 0.55, 0.35],
            [0.60, 0.35, 0.45, 0.18],
            [0.78, 0.60, 0.52, 0.28],
            [0.90, 0.82, 0.78, 0.70],
            [1.00, 1.00, 1.00, 1.00],
        ],
        dtype=np.float32,
    )


def _fallback_colormap(name: str, cdict: Optional[dict[str, Any]] = None) -> _FallbackColormap:
    anchors = _anchors_from_cdict(cdict) if cdict else _terrain_anchors()
    return _FallbackColormap(name, anchors)


def _anchors_from_cdict(cdict: dict[str, Any]) -> np.ndarray:
    channel_keys = ("red", "green", "blue")
    sample_positions = sorted({float(point[0]) for key in channel_keys for point in cdict.get(key, [])})
    if not sample_positions:
        return _terrain_anchors()

    anchors = np.zeros((len(sample_positions), 4), dtype=np.float32)
    anchors[:, 0] = np.asarray(sample_positions, dtype=np.float32)
    for channel_index, key in enumerate(channel_keys, start=1):
        points = cdict.get(key, [])
        if not points:
            anchors[:, channel_index] = anchors[:, 0]
            continue
        xs = np.array([float(point[0]) for point in points], dtype=np.float32)
        ys = np.array([float(point[1]) for point in points], dtype=np.float32)
        anchors[:, channel_index] = np.interp(anchors[:, 0], xs, ys)
    return anchors


def _legend_name(legend_path: Optional[Union[str, Path]]) -> str:
    raw = str(legend_path if legend_path is not None else DEFAULT_TOPO_LEGEND)
    if raw.startswith(":/"):
        return PurePosixPath(raw.removeprefix(":/")).stem
    return Path(raw).stem


def _read_legend_text(legend_path: Optional[Union[str, Path]]) -> Optional[str]:
    candidates = [legend_path] if legend_path is not None else [DEFAULT_TOPO_LEGEND, LEGENDS_DIR / "topography.txt"]
    for candidate in candidates:
        if candidate is None:
            continue

        raw = str(candidate)
        if raw.startswith(":/"):
            resource = QFile(raw)
            if not resource.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text):
                continue
            try:
                return bytes(resource.readAll()).decode("utf-8")
            finally:
                resource.close()

        path = Path(candidate)
        if path.is_file():
            return path.read_text(encoding="utf-8")

    return None


def get_named_cmap(name: str):
    if plt is not None:
        try:
            return plt.get_cmap(name)
        except Exception:
            pass
    return _fallback_colormap(name)


def load_topo_cmap(legend_path: Optional[Union[str, Path]] = None, *, name: Optional[str] = None):
    """Load the custom topographic colormap from legend data file."""
    cmap_name = name or _legend_name(legend_path)
    legend_text = _read_legend_text(legend_path)
    if legend_text is None:
        if plt is None:
            return _fallback_colormap(cmap_name)
        return plt.get_cmap("terrain")

    cdict = ast.literal_eval(legend_text)
    if plt is None:
        return _fallback_colormap(cmap_name, cdict if cdict else None)
    return LinearSegmentedColormap(cmap_name, cdict)
