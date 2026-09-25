"""Layer models for icosphere vertex data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class LayerLegend:
    """Reference to a render legend: display name + concrete colormap."""

    name: str
    colormap: Any

    def __post_init__(self) -> None:
        self.name = str(self.name)


@dataclass
class Layer:
    """A named data layer attached to each mesh vertex."""

    name: str
    values: np.ndarray
    description: str = ""
    visible: bool = True
    opacity: float = 100.0
    legend: Optional[LayerLegend] = None
    z_index: int = 0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1:
            raise ValueError(f"Layer values for {self.name!r} must be 1D, got shape {self.values.shape}.")
        self.opacity = float(self.opacity)
        if self.opacity < 0.0:
            self.opacity = 0.0
        elif self.opacity > 100.0:
            self.opacity = 100.0
        self.visible = bool(self.visible)
        self.z_index = int(self.z_index)

    @property
    def data(self) -> np.ndarray:
        return self.values

    @data.setter
    def data(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)
