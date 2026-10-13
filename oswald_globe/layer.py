"""Layer models for icosphere vertex data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
class DataLayer(ABC):
    """Abstract base class for a named dataset attached to each mesh vertex."""

    name: str
    description: str = ""
    visible: bool = True
    opacity: float = 100.0
    legend: Optional[LayerLegend] = None
    z_index: int = 0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.opacity = float(self.opacity)
        if self.opacity < 0.0:
            self.opacity = 0.0
        elif self.opacity > 100.0:
            self.opacity = 100.0
        self.visible = bool(self.visible)
        self.z_index = int(self.z_index)

    @property
    @abstractmethod
    def kind(self) -> str:
        """Stable identifier for the concrete dataset type."""

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """One value per vertex."""

    @values.setter
    @abstractmethod
    def values(self, values: np.ndarray) -> None:
        """Replace the per-vertex dataset contents."""

    @property
    def data(self) -> np.ndarray:
        return self.values

    @data.setter
    def data(self, values: np.ndarray) -> None:
        self.values = values


@dataclass
class ElevationLayer(DataLayer):
    """Continuous float elevation values for each mesh vertex."""

    _values: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64), repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.values = self._values

    @property
    def kind(self) -> str:
        return "elevation"

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"Layer values for {self.name!r} must be 1D, got shape {array.shape}.")
        self._values = array
