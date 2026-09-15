"""Desktop application for exploring the heightfield globe."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from threading import Event
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PySide6.QtCore import QDir, QObject, QSettings, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QColor, QImageReader, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from . import resources_rc  # noqa: F401
    from .colormap import DEFAULT_TOPO_LEGEND, LEGENDS_DIR, load_topo_cmap
    from .globe_viewer import GlobeViewer
    from .icosphere import IcosphereBuildCancelled, IcosphereGrid
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe import resources_rc  # noqa: F401
    from oswald_globe.colormap import DEFAULT_TOPO_LEGEND, LEGENDS_DIR, load_topo_cmap
    from oswald_globe.globe_viewer import GlobeViewer
    from oswald_globe.icosphere import IcosphereBuildCancelled, IcosphereGrid

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEIGHTFIELD = PACKAGE_ROOT / "data" / "heightfield.tif"
APP_SETTINGS = QSettings("Oswald Globe", "Oswald Globe")


def _empty_globe_elevation() -> np.ndarray:
    """Return a sea-level data array for a default globe without a heightfield."""
    return np.zeros((2, 2), dtype=np.float32)


def _generic_icosphere_mesh(level: int = 7) -> IcosphereGrid:
    """Build a uniform icosphere mesh at the requested subdivision level."""
    mesh = IcosphereGrid()
    mesh.subdivide_uniform(max(0, int(level)))
    return mesh


def load_elevation(heightfield: Path) -> np.ndarray:
    """Load the normalized heightfield using terrain elevation units."""
    with Image.open(heightfield) as image:
        heightfield_data = np.asarray(image, dtype=np.float32)
    return heightfield_data * 4096.0 - 1024.0


def _color_to_qcolor(color: Union[QColor, Tuple[float, float, float], str]) -> QColor:
    if isinstance(color, QColor):
        return QColor(color)
    if isinstance(color, str):
        qcolor = QColor(color)
        if not qcolor.isValid():
            raise ValueError(f"Invalid color value: {color!r}")
        return qcolor
    return QColor.fromRgbF(float(color[0]), float(color[1]), float(color[2]))


def _qcolor_to_rgb(color: QColor) -> Tuple[float, float, float]:
    return (color.redF(), color.greenF(), color.blueF())


class ColorButton(QPushButton):
    def __init__(self, color: Union[QColor, Tuple[float, float, float], str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = _color_to_qcolor(color)
        self.clicked.connect(self._choose_color)
        self._update_label()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: Union[QColor, Tuple[float, float, float], str]) -> None:
        self._color = _color_to_qcolor(color)
        self._update_label()

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Select color")
        if chosen.isValid():
            self._color = chosen
            self._update_label()

    def _update_label(self) -> None:
        self.setText(self._color.name().upper())
        self.setToolTip(self._color.name().upper())
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; color: white; padding: 4px 8px; }}"
        )


class GridSettingsTab(QWidget):
    def __init__(self, viewer: GlobeViewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._viewer = viewer

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Grid"), 0, 0)
        layout.addWidget(QLabel("Visible"), 0, 1)
        layout.addWidget(QLabel("Color"), 0, 2)
        layout.addWidget(QLabel("Opacity"), 0, 3)

        self._rows: Dict[str, Tuple[QCheckBox, ColorButton, QDoubleSpinBox]] = {}
        self._add_row(
            layout,
            1,
            "Icosphere",
            viewer.gl_widget.show_wireframe,
            viewer.gl_widget.mesh_color,
            viewer.gl_widget.mesh_opacity,
        )
        self._add_row(
            layout,
            2,
            "Graticule",
            viewer.gl_widget.is_graticule,
            viewer.gl_widget.graticule_color,
            viewer.gl_widget.graticule_opacity,
        )
        layout.setRowStretch(3, 1)

    def _add_row(
        self,
        layout: QGridLayout,
        row: int,
        name: str,
        visible: bool,
        color: Tuple[float, float, float],
        opacity: float,
    ) -> None:
        label = QLabel(name, self)
        checkbox = QCheckBox(self)
        checkbox.setChecked(bool(visible))
        color_button = ColorButton(QColor.fromRgbF(*color), self)
        opacity_spin = QDoubleSpinBox(self)
        opacity_spin.setRange(0.0, 100.0)
        opacity_spin.setDecimals(0)
        opacity_spin.setSuffix("%")
        opacity_spin.setSingleStep(5.0)
        opacity_spin.setValue(max(0.0, min(100.0, float(opacity) * 100.0)))

        self._rows[name.lower()] = (checkbox, color_button, opacity_spin)
        layout.addWidget(label, row, 0)
        layout.addWidget(checkbox, row, 1)
        layout.addWidget(color_button, row, 2)
        layout.addWidget(opacity_spin, row, 3)

    def settings(self) -> Dict[str, Dict[str, Union[bool, float, QColor]]]:
        result: Dict[str, Dict[str, Union[bool, float, QColor]]] = {}
        for name, (checkbox, color_button, opacity_spin) in self._rows.items():
            result[name] = {
                "visible": checkbox.isChecked(),
                "color": color_button.color(),
                "opacity": opacity_spin.value() / 100.0,
            }
        return result


class SettingsDialog(QDialog):
    def __init__(
        self,
        viewer: GlobeViewer,
        apply_callback: Callable[[Dict[str, Dict[str, Union[bool, float, QColor]]]], None],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 220)
        self._apply_callback = apply_callback

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.grid_tab = GridSettingsTab(viewer, self.tabs)
        self.tabs.addTab(self.grid_tab, "Grids")
        main_layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(self)
        self.apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self.ok_button = buttons.addButton("OK", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_button.clicked.connect(self.apply_changes)
        self.ok_button.clicked.connect(self.accept_changes)
        self.cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(buttons)

    def apply_changes(self) -> None:
        self._apply_callback(self.grid_tab.settings())

    def accept_changes(self) -> None:
        self.apply_changes()
        self.accept()


class PaintToolOptionsDialog(QDialog):
    _PAINT_MODES = ("Replace", "Add", "Subtract", "Minimum", "Maximum", "Multiply", "Average")

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        value: int = 50,
        mode: str = "Replace",
        radius_km: int = 100,
        falloff_percent: int = 50,
    ):
        super().__init__(parent)
        self.setWindowTitle("Paint Tool Options")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(False)
        self.resize(360, 220)

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.value_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.value_slider.setRange(0, 100)
        self.value_slider.setValue(int(value))

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(list(self._PAINT_MODES))
        self.mode_combo.setCurrentText(mode if mode in self._PAINT_MODES else self._PAINT_MODES[0])

        self.radius_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.radius_slider.setRange(0, 1000)
        self.radius_slider.setValue(int(radius_km))

        self.falloff_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.falloff_slider.setRange(0, 100)
        self.falloff_slider.setValue(int(falloff_percent))

        layout.addWidget(QLabel("Value"), 0, 0)
        layout.addWidget(self.value_slider, 0, 1)
        layout.addWidget(QLabel("Mode"), 1, 0)
        layout.addWidget(self.mode_combo, 1, 1)
        layout.addWidget(QLabel("Radius (km)"), 2, 0)
        layout.addWidget(self.radius_slider, 2, 1)
        layout.addWidget(QLabel("Falloff (%)"), 3, 0)
        layout.addWidget(self.falloff_slider, 3, 1)


class Tool(ABC):
    def __init__(self, parent: QWidget):
        self._parent = parent
        self._menu_action: Optional[QAction] = None
        self._options_dialog: Optional[QDialog] = None

    @property
    def menu_action(self) -> Optional[QAction]:
        return self._menu_action

    @property
    def options_dialog(self) -> Optional[QDialog]:
        return self._options_dialog

    @abstractmethod
    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        raise NotImplementedError

    def create_options_dialog(self) -> Optional[QDialog]:
        return None

    def show_options_dialog(self) -> None:
        if self._options_dialog is None:
            self._options_dialog = self.create_options_dialog()
            if self._options_dialog is not None:
                self._options_dialog.destroyed.connect(self._on_options_dialog_destroyed)
        if self._options_dialog is None:
            return
        self._options_dialog.show()
        self._options_dialog.raise_()
        self._options_dialog.activateWindow()

    @Slot()
    def _on_options_dialog_destroyed(self) -> None:
        self._options_dialog = None

    def dispose_options_dialog(self) -> None:
        if self._options_dialog is None:
            return
        dialog = self._options_dialog
        self._options_dialog = None
        dialog.close()
        dialog.deleteLater()


class NavigateTool(Tool):
    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction("Navigate", self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action


class PaintTool(Tool):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._value = 50
        self._mode = "Replace"
        self._radius_km = 100
        self._falloff_percent = 50

    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction("Paint", self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action

    def create_options_dialog(self) -> Optional[QDialog]:
        return PaintToolOptionsDialog(
            self._parent,
            value=self._value,
            mode=self._mode,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            self._value = int(self._options_dialog.value_slider.value())
            self._mode = str(self._options_dialog.mode_combo.currentText())
            self._radius_km = int(self._options_dialog.radius_slider.value())
            self._falloff_percent = int(self._options_dialog.falloff_slider.value())
        super().dispose_options_dialog()

    def value(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.value_slider.value())
        return self._value

    def mode(self) -> str:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return str(self._options_dialog.mode_combo.currentText())
        return self._mode

    def radius_km(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.radius_slider.value())
        return self._radius_km

    def falloff_percent(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.falloff_slider.value())
        return self._falloff_percent


class IcosphereProgressDialog(QDialog):
    cancelRequested = Signal()
    _STAGES = (
        ("loading-heightfield", "Loading heightfield file"),
        ("creating-faces", "Creating icosphere faces"),
        ("balancing-faces", "Balancing icosphere faces"),
        ("populating-faces", "Populating globe data"),
        ("displaying-globe", "Displaying globe"),
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Loading Heightfield")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._stage_labels: Dict[str, QLabel] = {}
        self._stage_bars: Dict[str, QProgressBar] = {}

        layout = QVBoxLayout(self)
        for phase, title in self._STAGES:
            row_layout = QVBoxLayout()
            title_layout = QHBoxLayout()

            title_label = QLabel(title, self)
            state_label = QLabel("Pending", self)
            title_layout.addWidget(title_label)
            title_layout.addStretch()
            title_layout.addWidget(state_label)

            progress_bar = QProgressBar(self)
            progress_bar.setRange(0, 1)
            progress_bar.setValue(0)
            progress_bar.setFormat("Pending")

            row_layout.addLayout(title_layout)
            row_layout.addWidget(progress_bar)
            layout.addLayout(row_layout)

            self._stage_labels[phase] = state_label
            self._stage_bars[phase] = progress_bar

        buttons = QDialogButtonBox(self)
        self._cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel_button.clicked.connect(self.cancel)
        layout.addWidget(buttons)

    def cancel(self) -> None:
        self.cancelRequested.emit()

    def update_progress(self, phase: str, done: int, total: int) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        if total <= 0:
            progress_bar.setRange(0, 0)
            progress_bar.setFormat("Working...")
            state_label.setText("Running")
            return
        maximum = max(1, int(total))
        progress_bar.setRange(0, maximum)
        progress_bar.setValue(min(int(done), maximum))
        progress_bar.setFormat(f"{done:,} / {maximum:,}")
        if done >= maximum:
            state_label.setText("Done")
        elif done > 0:
            state_label.setText("Running")
        else:
            state_label.setText("Starting")

    def start_stage(self, phase: str, total: int = 1) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        maximum = max(1, int(total))
        progress_bar.setRange(0, maximum)
        progress_bar.setValue(0)
        progress_bar.setFormat(f"0 / {maximum:,}")
        state_label.setText("Running")

    def finish_stage(self, phase: str, total: int = 1) -> None:
        self.update_progress(phase, total, total)

    def set_stage_busy(self, phase: str) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        progress_bar.setRange(0, 0)
        progress_bar.setFormat("Working...")
        state_label.setText("Running")

    def show_cancelling(self) -> None:
        self._cancel_button.setEnabled(False)
        self.setWindowTitle("Cancelling Heightfield Load")


class IcosphereBuildWorker(QObject):
    progressChanged = Signal(str, int, int)
    completed = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, elevation: np.ndarray, min_level: int, max_level: int, threshold: float):
        super().__init__()
        self._elevation = np.asarray(elevation, dtype=np.float32)
        self._min_level = min_level
        self._max_level = max_level
        self._threshold = threshold
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            grid = IcosphereGrid.from_equirectangular(
                self._elevation,
                min_level=self._min_level,
                max_level=self._max_level,
                threshold=self._threshold,
                progress_callback=self.progressChanged.emit,
                is_cancelled=self._cancel_event.is_set,
            )
        except IcosphereBuildCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(exc)
            return

        self.completed.emit(grid)

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()


class GlobeMainWindow(QMainWindow):
    def __init__(
        self,
        heightfield: Optional[Path] = None,
        mesh_min_level: int = 2,
        mesh_max_level: int = 7,
        mesh_threshold: float = 100.0,
        width: int = 850,
        height: int = 900,
        settings: Optional[QSettings] = None,
    ):
        super().__init__()
        self._mesh_min_level = mesh_min_level
        self._mesh_max_level = mesh_max_level
        self._mesh_threshold = mesh_threshold
        self._save_path: Optional[Path] = None
        self._settings = settings or APP_SETTINGS
        self._legend_action_group: Optional[QActionGroup] = None
        self._tool_action_group: Optional[QActionGroup] = None
        self._tools: Dict[str, Tool] = {
            "navigate": NavigateTool(self),
            "paint": PaintTool(self),
        }
        self._active_tool: Optional[Tool] = None
        self._file_menu: Optional[QMenu] = None
        self._tools_menu: Optional[QMenu] = None
        self._view_menu: Optional[QMenu] = None
        self._current_legend = self._load_current_legend()
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[IcosphereBuildWorker] = None
        self._progress_dialog: Optional[IcosphereProgressDialog] = None
        self._pending_heightfield: Optional[Path] = None
        self._pending_elevation: Optional[np.ndarray] = None
        self._pending_viewer: Optional[GlobeViewer] = None
        self._heightfield: Optional[Path] = None

        self._set_heightfield(heightfield)
        self.resize(width, height)
        self.setMinimumSize(480, 540)
        self._build_menu_bar()

    @property
    def viewer(self) -> GlobeViewer:
        return self.centralWidget()  # type: ignore[return-value]

    def _build_menu_bar(self) -> None:
        self._file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_heightfield)
        self._file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_heightfield)
        self._file_menu.addAction(save_action)

        save_as_action = QAction("Save As...", self)
        save_as_action.triggered.connect(self.save_heightfield_as)
        self._file_menu.addAction(save_as_action)

        self._file_menu.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        self._file_menu.addAction(settings_action)

        self.menuBar().addMenu("Edit")

        self._tools_menu = self.menuBar().addMenu("Tools")
        self._tool_action_group = QActionGroup(self._tools_menu)
        self._tool_action_group.setExclusive(True)

        self._tools["navigate"].create_menu_action(
            self._tools_menu,
            self._tool_action_group,
            on_selected=lambda: self._set_active_tool("navigate"),
        )

        self._tools["paint"].create_menu_action(
            self._tools_menu,
            self._tool_action_group,
            on_selected=lambda: self._set_active_tool("paint"),
        )

        if self._tools["navigate"].menu_action is None:
            raise RuntimeError("Navigate tool menu action was not created.")
        self._tools["navigate"].menu_action.setChecked(True)
        self._set_active_tool("navigate")

        self._view_menu = self.menuBar().addMenu("View")
        self._legend_menu = self._view_menu.addMenu("Legend")
        self._legend_menu.aboutToShow.connect(self._populate_legend_menu)

    def _set_active_tool(self, tool_name: str) -> None:
        next_tool = self._tools.get(tool_name)
        if next_tool is None:
            raise ValueError(f"Unknown tool selection: {tool_name}")

        if self._active_tool is not None and self._active_tool is not next_tool:
            self._active_tool.dispose_options_dialog()
        self._active_tool = next_tool
        self._active_tool.show_options_dialog()

    def _legend_name(self, legend_source: Union[str, Path]) -> str:
        return Path(str(legend_source).removeprefix(":/")).stem

    def _available_legend_sources(self) -> list[str]:
        resource_dir = QDir(":/legends")
        resource_legends = resource_dir.entryList(["*.txt"], QDir.Filter.Files, QDir.SortFlag.Name)
        if resource_legends:
            return [f":/legends/{legend}" for legend in resource_legends]

        if LEGENDS_DIR.is_dir():
            return [str(path) for path in sorted(LEGENDS_DIR.glob("*.txt"))]

        return []

    def _default_legend_source(self) -> str:
        legends = self._available_legend_sources()
        if not legends:
            return DEFAULT_TOPO_LEGEND

        for legend in legends:
            if self._legend_name(legend) == "topography":
                return legend
        return legends[0]

    def _load_current_legend(self) -> str:
        stored = str(self._settings.value("legend/current", DEFAULT_TOPO_LEGEND))
        available = self._available_legend_sources()
        if stored in available:
            return stored
        return self._default_legend_source()

    def _ensure_current_legend(self) -> str:
        available = self._available_legend_sources()
        if self._current_legend in available:
            return self._current_legend
        self._current_legend = self._default_legend_source()
        return self._current_legend

    def _populate_legend_menu(self) -> None:
        self._legend_menu.clear()

        legends = self._available_legend_sources()
        if not legends:
            no_legends_action = self._legend_menu.addAction("No legends available")
            no_legends_action.setEnabled(False)
            self._legend_action_group = None
            return

        current_legend = self._ensure_current_legend()
        self._legend_action_group = QActionGroup(self._legend_menu)
        self._legend_action_group.setExclusive(True)
        for legend_source in legends:
            action = self._legend_menu.addAction(self._legend_name(legend_source))
            action.setCheckable(True)
            action.setChecked(legend_source == current_legend)
            action.triggered.connect(lambda checked=False, source=legend_source: self.set_legend(source))
            self._legend_action_group.addAction(action)

    def set_legend(self, legend_source: Union[str, Path]) -> None:
        legend_value = str(legend_source)
        if legend_value not in self._available_legend_sources():
            raise FileNotFoundError(f"Legend not found: {legend_value}")

        self._current_legend = legend_value
        self._settings.setValue("legend/current", legend_value)
        self.viewer.set_colormap(load_topo_cmap(legend_value, name=self._legend_name(legend_value)))

    def _start_heightfield_load(self, heightfield: Path, elevation: np.ndarray) -> None:
        if self._load_thread is not None:
            raise RuntimeError("A heightfield load is already in progress.")

        self._pending_heightfield = Path(heightfield).expanduser().resolve()
        self._pending_elevation = np.asarray(elevation, dtype=np.float32)
        if self._progress_dialog is None:
            self._progress_dialog = IcosphereProgressDialog(self)
        self._load_thread = QThread(self)
        self._load_worker = IcosphereBuildWorker(
            self._pending_elevation,
            min_level=self._mesh_min_level,
            max_level=self._mesh_max_level,
            threshold=self._mesh_threshold,
        )
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progressChanged.connect(self._progress_dialog.update_progress)
        self._load_worker.completed.connect(self._on_heightfield_load_completed)
        self._load_worker.failed.connect(self._on_heightfield_load_failed)
        self._load_worker.cancelled.connect(self._on_heightfield_load_cancelled)
        self._load_worker.completed.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_worker.cancelled.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_heightfield_load)
        self._progress_dialog.cancelRequested.connect(self._load_worker.cancel)
        self._progress_dialog.cancelRequested.connect(self._progress_dialog.show_cancelling)

        self._progress_dialog.show()
        self._load_thread.start()

    @Slot(object)
    def _on_heightfield_load_completed(self, grid: object) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.set_stage_busy("displaying-globe")
        if self._pending_heightfield is None or self._pending_elevation is None:
            raise RuntimeError("Heightfield load completed without pending input data.")
        self._set_heightfield(self._pending_heightfield, elevation=self._pending_elevation, mesh_grid=grid)
        self._pending_viewer = self.viewer
        self._pending_viewer.gl_widget.firstFrameRendered.connect(self._finish_heightfield_display)
        self._pending_viewer.gl_widget.update()
        QTimer.singleShot(0, self._pending_viewer.gl_widget.update)

    @Slot(object)
    def _on_heightfield_load_failed(self, exc: object) -> None:
        self._close_progress_dialog()
        if not isinstance(exc, Exception):
            raise RuntimeError(f"Heightfield load failed with unexpected error payload: {exc!r}")
        QMessageBox.critical(self, "Failed to load heightfield", str(exc))

    @Slot()
    def _on_heightfield_load_cancelled(self) -> None:
        self._close_progress_dialog()

    @Slot()
    def _finish_heightfield_display(self) -> None:
        if self._pending_viewer is None:
            return
        try:
            self._pending_viewer.gl_widget.firstFrameRendered.disconnect(self._finish_heightfield_display)
        except (RuntimeError, TypeError):
            pass
        if self._progress_dialog is not None:
            self._progress_dialog.finish_stage("displaying-globe")
        QTimer.singleShot(0, self._close_progress_dialog)

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is None:
            return
        self._progress_dialog.close()
        self._progress_dialog.deleteLater()
        self._progress_dialog = None
        self._pending_viewer = None

    @Slot()
    def _cleanup_heightfield_load(self) -> None:
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None
        if self._load_thread is not None:
            self._load_thread.deleteLater()
            self._load_thread = None
        self._pending_heightfield = None
        self._pending_elevation = None

    def _set_heightfield(
        self,
        heightfield: Optional[Path],
        *,
        elevation: Optional[np.ndarray] = None,
        mesh_grid: Optional[IcosphereGrid] = None,
    ) -> None:
        existing_settings = None
        if isinstance(self.centralWidget(), GlobeViewer):
            existing_settings = self._current_grid_settings()

        if heightfield is None:
            elevation_data = _empty_globe_elevation() if elevation is None else np.asarray(elevation, dtype=np.float32)
            view_title = "Sea level"
            mesh = mesh_grid if mesh_grid is not None else _generic_icosphere_mesh(self._mesh_max_level)
            viewer = GlobeViewer(
                elevation_data,
                cmap=load_topo_cmap(self._ensure_current_legend(), name=self._legend_name(self._current_legend)),
                responsive=True,
                title=view_title,
                mesh=mesh,
                mesh_min_level=self._mesh_min_level,
                mesh_max_level=self._mesh_max_level,
                mesh_threshold=self._mesh_threshold,
            )
            if existing_settings is not None:
                self._apply_grid_settings_to_viewer(viewer, existing_settings)
            else:
                self._load_grid_settings(viewer)

            old_viewer = self.centralWidget()
            self.setCentralWidget(viewer)
            if old_viewer is not None:
                old_viewer.deleteLater()

            self.setWindowTitle(view_title)
            self._heightfield = None
            self._save_path = None
            return

        heightfield = Path(heightfield).expanduser().resolve()
        if not heightfield.is_file():
            raise FileNotFoundError(f"Heightfield not found: {heightfield}")

        elevation_data = load_elevation(heightfield) if elevation is None else np.asarray(elevation, dtype=np.float32)
        viewer = GlobeViewer(
            elevation_data,
            cmap=load_topo_cmap(self._ensure_current_legend(), name=self._legend_name(self._current_legend)),
            responsive=True,
            title=heightfield.stem,
            mesh=mesh_grid if mesh_grid is not None else True,
            mesh_min_level=self._mesh_min_level,
            mesh_max_level=self._mesh_max_level,
            mesh_threshold=self._mesh_threshold,
        )

        if existing_settings is not None:
            self._apply_grid_settings_to_viewer(viewer, existing_settings)
        else:
            self._load_grid_settings(viewer)

        old_viewer = self.centralWidget()
        self.setCentralWidget(viewer)
        if old_viewer is not None:
            old_viewer.deleteLater()

        self.setWindowTitle(heightfield.stem)
        self._heightfield = heightfield
        self._save_path = None

    def _current_grid_settings(self) -> Dict[str, Dict[str, Union[bool, float, QColor]]]:
        viewer = self.viewer
        return {
            "icosphere": {
                "visible": viewer.gl_widget.show_wireframe,
                "color": self._grid_color("icosphere", QColor.fromRgbF(*viewer.gl_widget.mesh_color)),
                "opacity": self._grid_opacity("icosphere", viewer.gl_widget.mesh_opacity),
            },
            "graticule": {
                "visible": viewer.gl_widget.is_graticule,
                "color": self._grid_color("graticule", QColor.fromRgbF(*viewer.gl_widget.graticule_color)),
                "opacity": self._grid_opacity("graticule", viewer.gl_widget.graticule_opacity),
            },
        }

    def _grid_color(self, grid: str, default: QColor) -> QColor:
        color = QColor(str(self._settings.value(f"grids/{grid}/color", default.name())))
        return color if color.isValid() else default

    def _grid_opacity(self, grid: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(self._settings.value(f"grids/{grid}/opacity", default))))
        except (TypeError, ValueError):
            return default

    def _grid_visible(self, grid: str, default: bool) -> bool:
        value = self._settings.value(f"grids/{grid}/visible", default)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)

    def _load_grid_settings(self, viewer: GlobeViewer) -> None:
        self._apply_grid_settings_to_viewer(viewer, {
            "icosphere": {
                "visible": self._grid_visible("icosphere", viewer.gl_widget.show_wireframe),
                "color": self._grid_color("icosphere", QColor.fromRgbF(*viewer.gl_widget.mesh_color)),
                "opacity": self._grid_opacity("icosphere", viewer.gl_widget.mesh_opacity),
            },
            "graticule": {
                "visible": self._grid_visible("graticule", viewer.gl_widget.is_graticule),
                "color": self._grid_color("graticule", QColor.fromRgbF(*viewer.gl_widget.graticule_color)),
                "opacity": self._grid_opacity("graticule", viewer.gl_widget.graticule_opacity),
            },
        })

    def _save_grid_settings(self) -> None:
        settings = self._current_grid_settings()
        for grid in ("icosphere", "graticule"):
            self._settings.setValue(f"grids/{grid}/visible", settings[grid]["visible"])
            self._settings.setValue(f"grids/{grid}/color", settings[grid]["color"].name())
            self._settings.setValue(f"grids/{grid}/opacity", settings[grid]["opacity"])

    def _apply_grid_settings_to_viewer(
        self,
        viewer: GlobeViewer,
        settings: Dict[str, Dict[str, Union[bool, float, QColor]]],
    ) -> None:
        icosphere = settings["icosphere"]
        graticule = settings["graticule"]
        viewer.set_mesh_wireframe(
            bool(icosphere["visible"]),
            icosphere["color"],
            float(icosphere["opacity"]),
        )
        viewer.set_graticule(
            bool(graticule["visible"]),
            graticule["color"],
            float(graticule["opacity"]),
        )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.viewer, self.apply_grid_settings, self)
        dialog.exec()

    def apply_grid_settings(self, settings: Dict[str, Dict[str, Union[bool, float, QColor]]]) -> None:
        self._apply_grid_settings_to_viewer(self.viewer, settings)
        self._save_grid_settings()

    def closeEvent(self, event: QCloseEvent) -> None:
        for tool in self._tools.values():
            tool.dispose_options_dialog()
        if self._load_worker is not None:
            self._load_worker.cancel()
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()
        self._save_grid_settings()
        self._settings.sync()
        super().closeEvent(event)

    def _image_file_filter(self) -> str:
        extensions = sorted({bytes(fmt).decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()})
        patterns = " ".join(f"*.{extension}" for extension in extensions)
        return f"Image Files ({patterns});;All Files (*)"

    def _create_open_heightfield_dialog(self) -> QFileDialog:
        starting_dir = self._heightfield.parent if self._heightfield is not None else Path.cwd()
        dialog = QFileDialog(self, "Open Heightfield", str(starting_dir))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(self._image_file_filter())
        return dialog

    def open_heightfield(self) -> None:
        dialog = self._create_open_heightfield_dialog()
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return

        heightfield = Path(selected_files[0])
        progress_dialog = IcosphereProgressDialog(self)
        self._progress_dialog = progress_dialog
        progress_dialog.start_stage("loading-heightfield")
        progress_dialog.show()
        try:
            elevation = load_elevation(heightfield)
        except Exception as exc:
            progress_dialog.close()
            progress_dialog.deleteLater()
            self._progress_dialog = None
            QMessageBox.critical(self, "Failed to load heightfield", str(exc))
            return
        progress_dialog.finish_stage("loading-heightfield")
        self._start_heightfield_load(heightfield, elevation)

    def save_heightfield(self) -> None:
        if self._save_path is None:
            self.save_heightfield_as()
            return
        self._save_heightfield(self._save_path)

    def save_heightfield_as(self) -> None:
        default_path = (self._heightfield.with_suffix(".npy") if self._heightfield is not None else Path("sea_level.npy"))
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Heightfield",
            str(default_path),
            "NumPy Array (*.npy);;Compressed NumPy Array (*.npz);;TIFF (*.tif *.tiff)",
        )
        if file_name:
            self._save_path = Path(file_name)
            self._save_heightfield(self._save_path)

    def _save_heightfield(self, path: Path) -> None:
        path = Path(path)
        data = self.viewer.data
        suffix = path.suffix.lower()
        if suffix == ".npy":
            np.save(path, data)
        elif suffix == ".npz":
            np.savez_compressed(path, elevation=data)
        elif suffix in (".tif", ".tiff"):
            Image.fromarray(np.asarray(data, dtype=np.float32)).save(path)
        else:
            raise ValueError(f"Unsupported save format: {path.suffix}")


def create_window(
    heightfield: Optional[Path] = None,
    mesh_min_level: int = 2,
    mesh_max_level: int = 7,
    mesh_threshold: float = 100.0,
    width: int = 850,
    height: int = 900,
    settings: Optional[QSettings] = None,
) -> QMainWindow:
    """Create a native desktop window hosting the interactive 3D globe."""
    return GlobeMainWindow(
        heightfield=heightfield,
        mesh_min_level=mesh_min_level,
        mesh_max_level=mesh_max_level,
        mesh_threshold=mesh_threshold,
        width=width,
        height=height,
        settings=settings,
    )


def _configure_opengl_surface_format() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D heightfield desktop globe.")
    parser.add_argument("--heightfield", type=Path, default=None, help="Optional raster file to load. Defaults to a sea-level globe.")
    parser.add_argument("--mesh-min-level", type=int, default=2, help="Minimum icosphere subdivision level.")
    parser.add_argument("--mesh-max-level", type=int, default=7, help="Maximum icosphere subdivision level.")
    parser.add_argument("--mesh-threshold", type=float, default=100.0, help="Gradient threshold driving adaptive refinement.")
    parser.add_argument("--width", type=int, default=850, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Window height in pixels.")
    args = parser.parse_args()

    _configure_opengl_surface_format()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = create_window(
        args.heightfield,
        mesh_min_level=args.mesh_min_level,
        mesh_max_level=args.mesh_max_level,
        mesh_threshold=args.mesh_threshold,
        width=args.width,
        height=args.height,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
