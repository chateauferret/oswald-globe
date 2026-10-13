"""Global simplex-noise fill filter plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QDialogButtonBox, QWidget

from oswald_globe.algorithms.apply_value import ApplyValue
from oswald_globe.algorithms.geo_simplex_noise import GeoSimplexNoise
from oswald_globe.icosphere_grid import IcosphereGrid
from oswald_globe.tools.noise_tool import NoiseToolOptionsDialog

from .filter import Filter

__all__ = ["NoiseFilter", "NoiseFilterCommand", "NoiseFilterOptionsDialog"]


class NoiseFilterOptionsDialog(NoiseToolOptionsDialog):
    applyRequested = Signal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        minimum_value: float = -32767.0,
        maximum_value: float = 32767.0,
        frequency: float = 1.0,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        octaves: int = 6,
        seed: int = 0,
    ):
        super().__init__(
            parent,
            title="Noise Filter Options",
            minimum_value=minimum_value,
            maximum_value=maximum_value,
            frequency=frequency,
            persistence=persistence,
            lacunarity=lacunarity,
            octaves=octaves,
            seed=seed,
            include_radius=False,
            include_falloff=False,
        )
        buttons = QDialogButtonBox(self)
        self._apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.close)
        self._apply_button.clicked.connect(self._apply_requested)
        self.set_action_widget(buttons)

    def accept(self) -> None:
        self.applyRequested.emit()
        self.close()

    def _apply_requested(self) -> None:
        self.applyRequested.emit()


@dataclass(slots=True)
class NoiseFilterCommand:
    """Fill the mesh-backed elevation layer from simplex noise values."""

    mesh_grid: IcosphereGrid
    minimum_value: float
    maximum_value: float
    frequency: float
    persistence: float
    lacunarity: float
    octaves: int
    seed: int
    apply_vertex_values: Callable[[np.ndarray, np.ndarray], None]
    vertex_indices: np.ndarray = field(init=False, repr=False)
    before_values: np.ndarray = field(init=False, repr=False)
    after_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertex_indices = np.arange(self.mesh_grid.vertex_count(), dtype=np.intp)
        layer = self.mesh_grid.get_layer("elevation")
        self.before_values = layer.values.astype(np.float64, copy=True)
        selection_values = self.mesh_grid.selection_values.astype(np.float64, copy=True)
        if not np.any(selection_values > 0.0):
            selection_values = np.ones_like(self.before_values, dtype=np.float64)
        lat_deg, lon_deg = self.mesh_grid.vertex_lat_lon()
        noise_values = GeoSimplexNoise.sample_many(
            lat_deg,
            lon_deg,
            minimum_value=self.minimum_value,
            maximum_value=self.maximum_value,
            frequency=self.frequency,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            octaves=self.octaves,
            seed=self.seed,
        )
        self.after_values = ApplyValue.blend(self.before_values, noise_values, weights=selection_values)

    def redo(self) -> None:
        self.apply_vertex_values(self.vertex_indices, self.after_values)

    def undo(self) -> None:
        self.apply_vertex_values(self.vertex_indices, self.before_values)


class NoiseFilter(Filter):
    _menu_label = "Noise"

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._minimum_value = -32767.0
        self._maximum_value = 32767.0
        self._frequency = 1.0
        self._persistence = 0.5
        self._lacunarity = 2.0
        self._octaves = 6
        self._seed = 0

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

    def create_options_dialog(self) -> NoiseFilterOptionsDialog:
        dialog = NoiseFilterOptionsDialog(
            self._parent,
            minimum_value=self._minimum_value,
            maximum_value=self._maximum_value,
            frequency=self._frequency,
            persistence=self._persistence,
            lacunarity=self._lacunarity,
            octaves=self._octaves,
            seed=self._seed,
        )
        dialog.applyRequested.connect(self._apply_from_dialog)
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

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            self._minimum_value = float(self._options_dialog.minimum_value_spin.value())
            self._maximum_value = float(self._options_dialog.maximum_value_spin.value())
            self._frequency = float(self._options_dialog.frequency_spin.value())
            self._persistence = float(self._options_dialog.persistence_spin.value())
            self._lacunarity = float(self._options_dialog.lacunarity_spin.value())
            self._octaves = int(self._options_dialog.octaves_spin.value())
            self._seed = int(self._options_dialog.seed_spin.value())
        super().dispose_options_dialog()

    def minimum_value(self) -> float:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return float(self._options_dialog.minimum_value_spin.value())
        return self._minimum_value

    def maximum_value(self) -> float:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return float(self._options_dialog.maximum_value_spin.value())
        return self._maximum_value

    def frequency(self) -> float:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return float(self._options_dialog.frequency_spin.value())
        return self._frequency

    def persistence(self) -> float:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return float(self._options_dialog.persistence_spin.value())
        return self._persistence

    def lacunarity(self) -> float:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return float(self._options_dialog.lacunarity_spin.value())
        return self._lacunarity

    def octaves(self) -> int:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return int(self._options_dialog.octaves_spin.value())
        return self._octaves

    def seed(self) -> int:
        if isinstance(self._options_dialog, NoiseFilterOptionsDialog):
            return int(self._options_dialog.seed_spin.value())
        return self._seed

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
            raise RuntimeError("Noise filter requires a mesh-backed globe.")

        command = NoiseFilterCommand(
            mesh_grid=mesh_grid,
            minimum_value=self.minimum_value(),
            maximum_value=self.maximum_value(),
            frequency=self.frequency(),
            persistence=self.persistence(),
            lacunarity=self.lacunarity(),
            octaves=self.octaves(),
            seed=self.seed(),
            apply_vertex_values=gl_widget.apply_mesh_vertex_values,
        )
        undo_stack = getattr(gl_widget, "_undo_stack", None)
        if undo_stack is None:
            command.redo()
            return
        undo_stack.push(command)
