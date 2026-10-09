"""Paint tool configuration and menu action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtWidgets import QWidget

from oswald_globe.brush_paint_command import BrushCommand
from oswald_globe.icosphere_grid import IcosphereGrid

from .tool import BrushTool
from .tool_options_dialog import ToolOptionsDialog

__all__ = ["BrushPaintCommand", "PaintTool", "PaintToolOptionsDialog"]


class PaintToolOptionsDialog(ToolOptionsDialog):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        value: int = 0,
        mode: str = "Replace",
        radius_km: int = 100,
        falloff_percent: int = 50,
        include_radius: bool = True,
        include_falloff: bool = True,
    ):
        super().__init__(
            parent,
            title="Paint Tool Options",
            value=value,
            mode=mode,
            radius_km=radius_km,
            falloff_percent=falloff_percent,
            include_value=True,
            include_mode=True,
            include_radius=include_radius,
            include_falloff=include_falloff,
        )


@dataclass(slots=True)
class BrushPaintCommand(BrushCommand):
    """Paint a brush stroke onto the mesh-backed elevation layer."""

    paint_value: int
    paint_mode: str
    apply_vertex_values: Callable[[np.ndarray, np.ndarray], None]

    def _apply(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.apply_vertex_values(indices, values)

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

    def _capture_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class PaintTool(BrushTool):
    _menu_label = "Paint"

    def __init__(self, parent: QWidget, on_brush_changed: Callable[[], None]):
        super().__init__(parent, on_brush_changed)
        self._value = 0
        self._mode = "Replace"

    def create_options_dialog(self) -> PaintToolOptionsDialog:
        dialog = PaintToolOptionsDialog(
            self._parent,
            value=self._value,
            mode=self._mode,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            self._value = int(self._options_dialog.value_slider.value())
            self._mode = str(self._options_dialog.mode_combo.currentText())
        super().dispose_options_dialog()

    def value(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.value_slider.value())
        return self._value

    def mode(self) -> str:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return str(self._options_dialog.mode_combo.currentText())
        return self._mode

    def create_brush_command(self, payload: Dict[str, Any], gl_widget: Any) -> BrushPaintCommand:
        mesh_grid = gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Paint tool requires a mesh-backed globe.")

        return BrushPaintCommand(
            target_lat_deg=float(payload["target_lat_deg"]),
            target_lon_deg=float(payload["target_lon_deg"]),
            radius_km=float(payload["radius_km"]),
            falloff_percent=float(payload["falloff_percent"]),
            paint_value=self.value(),
            paint_mode=self.mode(),
            mesh_grid=mesh_grid,
            apply_vertex_values=gl_widget.apply_mesh_vertex_values,
        )
