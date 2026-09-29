"""Brush paint command with undo/redo support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from oswald_globe.icosphere_grid import IcosphereGrid


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

    def _capture_brush_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inner_radius, outer_radius, angular_distances, vertex_indices = self._capture_vertex_distances(
            self.mesh_grid,
            self.target_lat_deg,
            self.target_lon_deg,
            self.radius_km,
            self.falloff_percent,
        )

        layer = self.mesh_grid.get_layer("elevation")
        before_values = layer.values[vertex_indices].astype(np.float64, copy=True)
        after_values = before_values.copy()
        if vertex_indices.size == 0:
            return vertex_indices, before_values, after_values

        paint_value = float(self.paint_value)
        operated_values = self._apply_paint_mode(before_values, paint_value, self.paint_mode)
        brush_values = before_values.copy()
        if inner_radius >= outer_radius - 1e-12:
            brush_values = operated_values
        else:
            affected_distances = angular_distances[vertex_indices]
            inner_mask = affected_distances <= inner_radius + 1e-12
            brush_values[inner_mask] = operated_values[inner_mask]

            transition_mask = ~inner_mask
            if np.any(transition_mask):
                blend = (affected_distances[transition_mask] - inner_radius) / (outer_radius - inner_radius)
                brush_values[transition_mask] = (
                    (1.0 - blend) * operated_values[transition_mask] + blend * before_values[transition_mask]
                )

        if not self.mesh_grid.has_selection:
            return vertex_indices, before_values, brush_values

        selection_values = self.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values[vertex_indices]
        after_values = before_values + selection_values * (brush_values - before_values)
        return vertex_indices, before_values, after_values

    def redo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_vertex_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_vertex_values(self.vertex_indices, self.before_values)


@dataclass(slots=True)
class SelectionBrushCommand:
    """Paint a soft-selection mask onto the mesh-backed selection layer."""

    target_lat_deg: float
    target_lon_deg: float
    radius_km: float
    falloff_percent: float
    replace_existing: bool
    erase_selection: bool
    mesh_grid: IcosphereGrid
    apply_selection_values: Callable[[np.ndarray, np.ndarray], None]
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)
    after_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices, self.before_values, self.after_values = self._capture_selection_values()

    def _capture_selection_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inner_radius, outer_radius, angular_distances, vertex_indices = BrushPaintCommand._capture_vertex_distances(
            self.mesh_grid,
            self.target_lat_deg,
            self.target_lon_deg,
            self.radius_km,
            self.falloff_percent,
        )

        layer = self.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        source_values = layer.values.astype(np.float64, copy=True)
        if self.replace_existing:
            before_values = source_values
            after_values = np.zeros_like(before_values)
        else:
            before_values = source_values[vertex_indices].copy()
            after_values = before_values.copy()
        if vertex_indices.size == 0:
            if self.replace_existing:
                return np.arange(self.mesh_grid.vertex_count(), dtype=np.intp), before_values, after_values
            return vertex_indices, before_values, after_values

        if self.erase_selection:
            weights = np.zeros(vertex_indices.shape, dtype=np.float64)
        else:
            affected_distances = angular_distances[vertex_indices]
            if inner_radius >= outer_radius - 1e-12:
                weights = np.ones(vertex_indices.shape, dtype=np.float64)
            else:
                weights = np.clip(
                    (outer_radius - affected_distances) / (outer_radius - inner_radius),
                    0.0,
                    1.0,
                )
                weights[affected_distances <= inner_radius + 1e-12] = 1.0
        if self.replace_existing:
            after_values[vertex_indices] = weights
            return np.arange(self.mesh_grid.vertex_count(), dtype=np.intp), before_values, after_values

        if self.erase_selection:
            after_values = weights
        else:
            after_values = np.maximum(before_values, weights)
        return vertex_indices, before_values, after_values

    def redo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_selection_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        if self.vertex_indices.size == 0:
            return
        self.apply_selection_values(self.vertex_indices, self.before_values)


@dataclass(slots=True)
class SelectionLayerCommand:
    """Apply a bulk update to the mesh-backed selection layer."""

    mesh_grid: IcosphereGrid
    after_values: np.ndarray
    apply_selection_values: Callable[[np.ndarray, np.ndarray], None]
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices = np.arange(self.mesh_grid.vertex_count(), dtype=np.intp)
        self.before_values = self.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values.astype(
            np.float64,
            copy=True,
        )
        self.after_values = np.clip(np.asarray(self.after_values, dtype=np.float64), 0.0, 1.0)
        if self.after_values.shape != self.before_values.shape:
            raise ValueError(
                "Selection layer update must contain one value per mesh vertex, "
                f"got {self.after_values.shape} for {self.before_values.shape}."
            )

    def redo(self) -> None:
        self.apply_selection_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        self.apply_selection_values(self.vertex_indices, self.before_values)
