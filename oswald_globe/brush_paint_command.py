"""Brush paint command with undo/redo support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from oswald_globe.icosphere import IcosphereGrid


@dataclass(slots=True)
class BrushPaintCommand:
    """Paint a brush stroke onto the mesh-backed elevation layer."""

    target_lat_deg: float
    target_lon_deg: float
    radius_km: float
    falloff_percent: float
    paint_value: int
    paint_mode: str
    mesh_grid: IcosphereGrid
    apply_vertex_values: Callable[[np.ndarray, np.ndarray], None]
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)
    after_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices, self.before_values, self.after_values = self._capture_brush_values()

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

    @staticmethod
    def _apply_paint_mode(existing_values: np.ndarray, paint_value: float, paint_mode: str) -> np.ndarray:
        mode = str(paint_mode)
        if mode == "Replace":
            return np.full(existing_values.shape, paint_value, dtype=np.float64)
        if mode == "Add":
            return existing_values + paint_value
        if mode == "Subtract":
            return existing_values - paint_value
        if mode == "Minimum":
            return np.minimum(existing_values, paint_value)
        if mode == "Maximum":
            return np.maximum(existing_values, paint_value)
        if mode == "Multiply":
            return existing_values * paint_value
        if mode == "Average":
            return (existing_values + paint_value) / 2.0
        raise ValueError(f"Unsupported paint mode: {paint_mode!r}")

    def _capture_brush_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inner_radius, outer_radius = self._brush_radii(self.radius_km, self.falloff_percent)
        center_xyz = self._latlon_to_xyz(self.target_lat_deg, self.target_lon_deg)
        angular_distances = np.arccos(np.clip(self.mesh_grid.vertices @ center_xyz, -1.0, 1.0))
        vertex_indices = np.flatnonzero(angular_distances <= outer_radius + 1e-9)

        layer = self.mesh_grid.get_layer("elevation")
        before_values = layer.values[vertex_indices].astype(np.float64, copy=True)
        after_values = before_values.copy()
        if vertex_indices.size == 0:
            return vertex_indices, before_values, after_values

        paint_value = float(self.paint_value)
        operated_values = self._apply_paint_mode(before_values, paint_value, self.paint_mode)
        if inner_radius >= outer_radius - 1e-12:
            after_values = operated_values
            return vertex_indices, before_values, after_values

        affected_distances = angular_distances[vertex_indices]
        inner_mask = affected_distances <= inner_radius + 1e-12
        after_values[inner_mask] = operated_values[inner_mask]

        transition_mask = ~inner_mask
        if np.any(transition_mask):
            blend = (affected_distances[transition_mask] - inner_radius) / (outer_radius - inner_radius)
            after_values[transition_mask] = (
                (1.0 - blend) * operated_values[transition_mask] + blend * before_values[transition_mask]
            )

        return vertex_indices, before_values, after_values

    def redo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_vertex_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_vertex_values(self.vertex_indices, self.before_values)
