"""Selection tool configuration and menu action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtWidgets import QWidget

from oswald_globe.brush_paint_command import BrushCommand
from oswald_globe.icosphere_grid import IcosphereGrid

from .tool import BrushTool
from .tool_options_dialog import ToolOptionsDialog

__all__ = ["SelectTool", "SelectToolOptionsDialog", "SelectionBrushCommand", "SelectionLayerCommand"]


class SelectToolOptionsDialog(ToolOptionsDialog):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        radius_km: int = 100,
        falloff_percent: int = 50,
        include_radius: bool = True,
        include_falloff: bool = True,
    ):
        super().__init__(
            parent,
            title="Select Tool Options",
            radius_km=radius_km,
            falloff_percent=falloff_percent,
            include_value=False,
            include_mode=False,
            include_radius=include_radius,
            include_falloff=include_falloff,
        )


@dataclass(slots=True)
class SelectionBrushCommand(BrushCommand):
    """Paint a soft-selection mask onto the mesh-backed selection layer."""

    replace_existing: bool
    erase_selection: bool
    apply_selection_values: Callable[[np.ndarray, np.ndarray], None]

    def _apply(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.apply_selection_values(indices, values)

    def _capture_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inner_radius, outer_radius, angular_distances, vertex_indices = self._capture_vertex_distances(
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


class SelectTool(BrushTool):
    _menu_label = "Select"

    def create_options_dialog(self) -> SelectToolOptionsDialog:
        dialog = SelectToolOptionsDialog(
            self._parent,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    def __init__(self, parent: QWidget, on_brush_changed: Callable[[], None]):
        super().__init__(parent, on_brush_changed)

    def create_brush_command(self, payload: Dict[str, Any], gl_widget: Any) -> SelectionBrushCommand:
        mesh_grid = gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Select tool requires a mesh-backed globe.")

        return SelectionBrushCommand(
            target_lat_deg=float(payload["target_lat_deg"]),
            target_lon_deg=float(payload["target_lon_deg"]),
            radius_km=float(payload["radius_km"]),
            falloff_percent=float(payload["falloff_percent"]),
            replace_existing=bool(payload.get("replace_existing", False)),
            erase_selection=bool(payload.get("erase_selection", False)),
            mesh_grid=mesh_grid,
            apply_selection_values=gl_widget.apply_mesh_selection_values,
        )
