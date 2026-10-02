"""Global fill filter plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QWidget

from oswald_globe.icosphere_grid import IcosphereGrid
from oswald_globe.tools.paint_tool import PaintToolOptionsDialog

from .filter import Filter

__all__ = ["FillFilter", "FillFilterCommand", "FillFilterOptionsDialog"]


class FillFilterOptionsDialog(PaintToolOptionsDialog):
    applyRequested = Signal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        value: int = 0,
        mode: str = "Replace",
        radius_km: int = 100,
        falloff_percent: int = 50,
    ):
        super().__init__(
            parent,
            value=value,
            mode=mode,
            radius_km=radius_km,
            falloff_percent=falloff_percent,
            include_radius=False,
            include_falloff=False,
        )
        self.setWindowTitle("Fill Filter Options")
        buttons = QDialogButtonBox(self)
        self._apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._apply_button.clicked.connect(self._apply_requested)
        self.layout().addWidget(buttons, self.layout().rowCount(), 0, 1, 3)

    def _apply_requested(self) -> None:
        self.applyRequested.emit()


@dataclass(slots=True)
class FillFilterCommand:
    """Apply a whole-selection fill operation to the mesh-backed elevation layer."""

    mesh_grid: IcosphereGrid
    fill_value: float
    fill_mode: str
    apply_vertex_values: Callable[[np.ndarray, np.ndarray], None]
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)
    after_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices = np.arange(self.mesh_grid.vertex_count(), dtype=np.intp)
        layer = self.mesh_grid.get_layer("elevation")
        self.before_values = layer.values.astype(np.float64, copy=True)
        selection_values = self.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values.astype(np.float64, copy=True)
        if not np.any(selection_values > 0.0):
            selection_values = np.ones_like(self.before_values, dtype=np.float64)
        transformed_values = self._apply_fill_mode(self.before_values, float(self.fill_value), str(self.fill_mode))
        self.after_values = self.before_values + selection_values * (transformed_values - self.before_values)

    @staticmethod
    def _apply_fill_mode(existing_values: np.ndarray, fill_value: float, fill_mode: str) -> np.ndarray:
        mode = str(fill_mode)
        if mode == "Replace":
            return np.full(existing_values.shape, fill_value, dtype=np.float64)
        if mode == "Add":
            return existing_values + fill_value
        if mode == "Subtract":
            return existing_values - fill_value
        if mode == "Minimum":
            return np.minimum(existing_values, fill_value)
        if mode == "Maximum":
            return np.maximum(existing_values, fill_value)
        if mode == "Multiply":
            return existing_values * fill_value
        if mode == "Average":
            return (existing_values + fill_value) / 2.0
        raise ValueError(f"Unsupported fill mode: {fill_mode!r}")

    def redo(self) -> None:
        self.apply_vertex_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        self.apply_vertex_values(self.vertex_indices, self.before_values)


class FillFilter(Filter):
    _menu_label = "Fill"

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._value = 0
        self._mode = "Replace"

    def create_menu_action(
        self,
        filters_menu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction(self._menu_label, self._parent)
        action.triggered.connect(lambda checked=False: on_selected())
        filters_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = FillFilterOptionsDialog(
            self._parent,
            value=self._value,
            mode=self._mode,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.applyRequested.connect(self._apply_from_dialog)
        dialog.accepted.connect(self._apply_from_dialog)
        return dialog

    @Slot()
    def _apply_from_dialog(self) -> None:
        viewer = getattr(self._parent, "viewer", None)
        if viewer is None:
            return
        gl_widget = getattr(viewer, "gl_widget", None)
        if gl_widget is None:
            return
        self.apply(gl_widget)

    def _notify_brush_changed(self, _value: int) -> None:
        pass

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, FillFilterOptionsDialog):
            self._value = int(self._options_dialog.value_slider.value())
            self._mode = str(self._options_dialog.mode_combo.currentText())
        super().dispose_options_dialog()

    def value(self) -> int:
        if isinstance(self._options_dialog, FillFilterOptionsDialog):
            return int(self._options_dialog.value_slider.value())
        return self._value

    def mode(self) -> str:
        if isinstance(self._options_dialog, FillFilterOptionsDialog):
            return str(self._options_dialog.mode_combo.currentText())
        return self._mode

    def apply(self, gl_widget: Any | None = None) -> None:
        if gl_widget is None:
            viewer = getattr(self._parent, "viewer", None)
            if viewer is None:
                return
            gl_widget = getattr(viewer, "gl_widget", None)
        if gl_widget is None:
            return

        mesh_grid = gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Fill filter requires a mesh-backed globe.")

        command = FillFilterCommand(
            mesh_grid=mesh_grid,
            fill_value=float(self.value()),
            fill_mode=self.mode(),
            apply_vertex_values=gl_widget.apply_mesh_vertex_values,
        )
        undo_stack = getattr(gl_widget, "_undo_stack", None)
        if undo_stack is None:
            command.redo()
            return
        undo_stack.push(command)
