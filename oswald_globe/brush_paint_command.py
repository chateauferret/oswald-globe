"""Brush-based mesh edit commands with undo/redo support."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from oswald_globe.icosphere_grid import IcosphereGrid


@dataclass(slots=True)
class BrushCommand(ABC):
    """Common base for undoable commands that paint a lat/lon brush stroke
    onto a mesh-backed layer.

    Subclasses implement `_capture_values` to compute the affected vertex
    indices and their before/after values, and `_apply` to write a set of
    values back onto their target layer.
    """

    target_lat_deg: float
    target_lon_deg: float
    radius_km: float
    falloff_percent: float
    mesh_grid: IcosphereGrid
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)
    after_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices, self.before_values, self.after_values = self._capture_values()

    @abstractmethod
    def _capture_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _apply(self, indices: np.ndarray, values: np.ndarray) -> None:
        raise NotImplementedError

    @staticmethod
    def _brush_radii(radius_km: float, falloff_percent: float) -> tuple[float, float]:
        outer_radius_deg = min(89.0, max(0.0, float(radius_km) / 111.0))
        outer_radius = math.radians(outer_radius_deg)
        inner_radius = outer_radius * min(100.0, max(0.0, float(falloff_percent))) / 100.0
        return inner_radius, outer_radius

    @staticmethod
    def _latlon_to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        cos_lat = math.cos(lat_rad)
        return np.array(
            [cos_lat * math.cos(lon_rad), cos_lat * math.sin(lon_rad), math.sin(lat_rad)],
            dtype=np.float64,
        )

    @classmethod
    def _capture_vertex_distances(
        cls,
        mesh_grid: IcosphereGrid,
        target_lat_deg: float,
        target_lon_deg: float,
        radius_km: float,
        falloff_percent: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        inner_radius, outer_radius = cls._brush_radii(radius_km, falloff_percent)
        center_xyz = cls._latlon_to_xyz(target_lat_deg, target_lon_deg)
        angular_distances = np.arccos(np.clip(mesh_grid.vertices @ center_xyz, -1.0, 1.0))
        vertex_indices = np.flatnonzero(angular_distances <= outer_radius + 1e-9)
        return inner_radius, outer_radius, angular_distances, vertex_indices

    def redo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self._apply(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self._apply(self.vertex_indices, self.before_values)


def __getattr__(name: str):
    if name == "BrushPaintCommand":
        from oswald_globe.tools.paint_tool import BrushPaintCommand

        return BrushPaintCommand
    if name == "SelectionBrushCommand":
        from oswald_globe.tools.select_tool import SelectionBrushCommand

        return SelectionBrushCommand
    if name == "SelectionLayerCommand":
        from oswald_globe.tools.select_tool import SelectionLayerCommand

        return SelectionLayerCommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BrushCommand", "BrushPaintCommand", "SelectionBrushCommand", "SelectionLayerCommand"]
