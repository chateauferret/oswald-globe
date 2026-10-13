"""Paint tool configuration and menu action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtWidgets import QWidget

from oswald_globe.algorithms.apply_value import ApplyValue
from oswald_globe.tools.brush_command import BrushCommand
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

        brush_weights = np.zeros(before_values.shape, dtype=np.float64)
        if inner_radius >= outer_radius - 1e-12:
            brush_weights.fill(1.0)
        else:
            affected_distances = angular_distances[vertex_indices]
            inner_mask = affected_distances <= inner_radius + 1e-12
            brush_weights[inner_mask] = 1.0

            transition_mask = ~inner_mask
            if np.any(transition_mask):
                brush_weights[transition_mask] = 1.0 - (
                    (affected_distances[transition_mask] - inner_radius) / (outer_radius - inner_radius)
                )

        if self.mesh_grid.has_selection:
            brush_weights *= self.mesh_grid.selection_values[vertex_indices]

        after_values = ApplyValue.apply(
            before_values,
            float(self.paint_value),
            self.paint_mode,
            weights=brush_weights,
        )
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
