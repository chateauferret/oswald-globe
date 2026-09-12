"""Colormap utilities for globe visualization."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:  # pragma: no cover - exercised in matplotlib-free environments
    plt = None
    LinearSegmentedColormap = Any  # type: ignore[assignment]

LEGENDS_DIR = Path(__file__).resolve().parent.parent / "legends"
DEFAULT_TOPO_LEGEND = LEGENDS_DIR / "topography.txt"


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


def _fallback_colormap(name: str) -> _FallbackColormap:
    if name.lower() == "topo":
        path = DEFAULT_TOPO_LEGEND
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                cdict = ast.literal_eval(f.read())
            return _FallbackColormap("topo", _anchors_from_cdict(cdict))
    return _FallbackColormap(name, _terrain_anchors())


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


def get_named_cmap(name: str):
    if plt is not None:
        try:
            return plt.get_cmap(name)
        except Exception:
            pass
    return _fallback_colormap(name)


def load_topo_cmap(legend_path: Optional[Path] = None):
    """Load the custom topographic colormap from legend data file."""
    path = Path(legend_path) if legend_path is not None else DEFAULT_TOPO_LEGEND
    if plt is None:
        if not path.is_file():
            return _fallback_colormap("terrain")
        with path.open("r", encoding="utf-8") as f:
            cdict = ast.literal_eval(f.read())
        return _fallback_colormap("topo") if not cdict else _FallbackColormap("topo", _anchors_from_cdict(cdict))

    if not path.is_file():
        return plt.get_cmap("terrain")

    with path.open("r", encoding="utf-8") as f:
        cdict = ast.literal_eval(f.read())
    return LinearSegmentedColormap("topo", cdict)
