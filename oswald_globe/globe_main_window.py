"""Desktop application for exploring the heightfield globe."""

from __future__ import annotations

import argparse
import sys
from threading import Event
from pathlib import Path
from typing import Dict, Optional, Union
from PySide6.QtCore import QSize
import numpy as np
from PIL import Image
from PySide6.QtCore import QDir, QObject, QSettings, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QColor, QIcon, QKeySequence, QLinearGradient, QPainter, QPixmap, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
    from_origin = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from . import resources_rc  # noqa: F401
    from .app_support import APP_SETTINGS, DEFAULT_HEIGHTFIELD, empty_globe_elevation, generic_icosphere_mesh, load_elevation
    from .colormap import DEFAULT_TOPO_LEGEND, LEGENDS_DIR, load_topo_cmap
    from .globe_viewer import GlobeViewer
    from .icosphere_grid import IcosphereGrid
    from .icosphere_build_worker import IcosphereBuildWorker
    from .icosphere_progress_dialog import IcosphereProgressDialog
    from .paint_tool import PaintTool
    from .project import Project
    from .settings_dialog import SettingsDialog
    from .tool import NavigateTool, Tool
    from .undo_stack import BrushPaintCommand, UndoStack
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe import resources_rc  # noqa: F401
    from oswald_globe.app_support import APP_SETTINGS, DEFAULT_HEIGHTFIELD, empty_globe_elevation, generic_icosphere_mesh, load_elevation
    from oswald_globe.colormap import DEFAULT_TOPO_LEGEND, LEGENDS_DIR, load_topo_cmap
    from oswald_globe.globe_viewer import GlobeViewer
    from oswald_globe.icosphere_grid import IcosphereGrid
    from oswald_globe.icosphere_build_worker import IcosphereBuildWorker
    from oswald_globe.icosphere_progress_dialog import IcosphereProgressDialog
    from oswald_globe.paint_tool import PaintTool
    from oswald_globe.project import Project
    from oswald_globe.settings_dialog import SettingsDialog
    from oswald_globe.tool import NavigateTool, Tool
    from oswald_globe.undo_stack import BrushPaintCommand, UndoStack


def _write_tiff_export(path: Path, raster: np.ndarray) -> None:
    clipped = np.clip(np.asarray(raster, dtype=np.float64), -32767.0, 32767.0)
    scaled = np.rint((clipped + 32767.0) * (65535.0 / 65534.0)).astype(np.uint16)
    if rasterio is not None and from_origin is not None:
        pixel_width = 360.0 / 4096.0
        pixel_height = 180.0 / 2048.0
        transform = from_origin(-180.0, 90.0, pixel_width, pixel_height)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2048,
            width=4096,
            count=1,
            dtype=np.uint16,
            crs="EPSG:4326",
            transform=transform,
            nodata=None,
        ) as dataset:
            dataset.write(scaled, 1)
        return

    Image.fromarray(scaled).save(path, format="TIFF")


def _load_tiff_import_raster(path: Path) -> np.ndarray:
    def _normalize_luminance(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            else:
                rgb = arr[:, :, :3].astype(np.float32, copy=False)
                max_channel_value = (
                    float(np.iinfo(arr.dtype).max)
                    if np.issubdtype(arr.dtype, np.integer)
                    else 1.0
                )
                if max_channel_value <= 0.0:
                    raise ValueError(f"Unsupported TIFF channel range for {path}.")
                rgb /= max_channel_value
                return np.clip(
                    0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2],
                    0.0,
                    1.0,
                )

        if arr.ndim != 2:
            raise ValueError(f"Expected TIFF raster to be 2D or RGB/RGBA, got shape {arr.shape}.")

        grayscale = arr.astype(np.float32, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            max_value = float(np.iinfo(arr.dtype).max)
        else:
            finite = grayscale[np.isfinite(grayscale)]
            max_value = float(np.nanmax(finite)) if finite.size else 0.0
            if max_value <= 0.0:
                max_value = 1.0
        if max_value <= 0.0:
            raise ValueError(f"Unsupported TIFF grayscale range for {path}.")
        return np.clip(grayscale / max_value, 0.0, 1.0)

    with Image.open(path) as image:
        source = image
        if image.mode == "P":
            source = image.convert("RGBA")
        normalized = _normalize_luminance(np.asarray(source))
        width, height = int(normalized.shape[1]), int(normalized.shape[0])
        if width <= 0 or height <= 0:
            raise ValueError(f"TIFF image has invalid size: {width}x{height}")
        if width != height * 2:
            if width / float(height) >= 2.0:
                new_size = (width, max(1, int(round(width / 2.0))))
            else:
                new_size = (max(1, int(round(height * 2.0))), height)
            resized = Image.fromarray(normalized.astype(np.float32), mode="F").resize(
                new_size,
                resample=Image.Resampling.BILINEAR,
            )
            normalized = np.asarray(resized, dtype=np.float32)

    return normalized * 65534.0 - 32767.0


class TiffExportWorker(QObject):
    progressMessageChanged = Signal(str)
    progressChanged = Signal(int, int)
    completed = Signal()
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, mesh_grid: IcosphereGrid, export_path: Path):
        super().__init__()
        self._mesh_grid = mesh_grid
        self._export_path = Path(export_path)
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            self.progressMessageChanged.emit("Rasterizing icosphere data")
            self.progressChanged.emit(0, 100)
            raster = self._rasterize_with_progress(height=2048, width=4096)
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            self.progressMessageChanged.emit("Writing TIFF file")
            self.progressChanged.emit(95, 100)
            _write_tiff_export(self._export_path, raster)
            self.progressChanged.emit(100, 100)
        except Exception as exc:
            self.failed.emit(exc)
            return
        self.completed.emit()

    def _rasterize_with_progress(self, *, height: int, width: int, chunk_rows: int = 64) -> np.ndarray:
        rows = np.arange(height)
        cols = np.arange(width)
        lat = 90.0 - (rows + 0.5) / height * 180.0
        lon = (cols + 0.5) / width * 360.0 - 180.0
        lon_grid = np.broadcast_to(lon, (chunk_rows, width))
        raster = np.empty((height, width), dtype=np.float64)
        total_rows = max(1, height)

        for start in range(0, height, chunk_rows):
            if self._cancel_event.is_set():
                break
            end = min(start + chunk_rows, height)
            chunk_height = end - start
            lat_chunk = lat[start:end]
            lat_grid = np.repeat(lat_chunk[:, None], width, axis=1)
            sampled = self._mesh_grid.sample(
                lat_grid.ravel(),
                lon_grid[:chunk_height].ravel(),
                k=8,
            ).reshape(chunk_height, width)
            raster[start:end, :] = sampled
            rasterize_progress = int(round((end / total_rows) * 95.0))
            self.progressChanged.emit(rasterize_progress, 100)

        return raster

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()


class TiffImportWorker(QObject):
    progressMessageChanged = Signal(str)
    progressChanged = Signal(int, int)
    completed = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    _STAGE_LABELS = {
        "creating-faces": "Creating icosphere faces",
        "balancing-faces": "Balancing icosphere faces",
        "populating-faces": "Populating globe data",
    }

    def __init__(self, import_path: Path, *, min_level: int, max_level: int, threshold: float):
        super().__init__()
        self._import_path = Path(import_path)
        self._min_level = int(min_level)
        self._max_level = int(max_level)
        self._threshold = float(threshold)
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            self.progressMessageChanged.emit("Loading TIFF file")
            self.progressChanged.emit(0, 100)
            raster = _load_tiff_import_raster(self._import_path)
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return

            self.progressMessageChanged.emit("Mapping TIFF onto icosphere data")
            grid = IcosphereGrid.from_equirectangular(
                raster,
                min_level=self._min_level,
                max_level=self._max_level,
                threshold=self._threshold,
                progress_callback=self._on_grid_progress,
                is_cancelled=self._cancel_event.is_set,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return

            self.progressChanged.emit(100, 100)
        except Exception as exc:
            self.failed.emit(exc)
            return

        self.completed.emit((raster, grid))

    def _on_grid_progress(self, phase: str, done: int, total: int) -> None:
        self.progressMessageChanged.emit(self._STAGE_LABELS.get(phase, "Importing TIFF"))
        stage_starts = {
            "creating-faces": 10.0,
            "balancing-faces": 70.0,
            "populating-faces": 85.0,
        }
        stage_widths = {
            "creating-faces": 60.0,
            "balancing-faces": 15.0,
            "populating-faces": 15.0,
        }
        start = stage_starts.get(phase, 10.0)
        width = stage_widths.get(phase, 0.0)
        if total <= 0:
            fraction = 0.0 if done <= 0 else 1.0
        else:
            bounded_total = max(1, int(total))
            fraction = min(max(float(done) / bounded_total, 0.0), 1.0)
        percent = int(round(start + width * fraction))
        self.progressChanged.emit(percent, 100)

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()


class TiffExportProgressDialog(QDialog):
    cancelRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Exporting TIFF")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._status_label = QLabel("Preparing export", self)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")

        buttons = QDialogButtonBox(self)
        self._cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel_button.clicked.connect(self.cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(buttons)

    def cancel(self) -> None:
        self.cancelRequested.emit()

    def set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def update_progress(self, done: int, total: int) -> None:
        maximum = max(1, int(total))
        bounded = max(0, min(int(done), maximum))
        self._progress_bar.setRange(0, maximum)
        self._progress_bar.setValue(bounded)

    def show_cancelling(self) -> None:
        self._cancel_button.setEnabled(False)
        self._status_label.setText("Cancelling export")
        self.setWindowTitle("Cancelling TIFF Export")


class TiffImportProgressDialog(QDialog):
    cancelRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Importing TIFF")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._status_label = QLabel("Preparing import", self)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")

        buttons = QDialogButtonBox(self)
        self._cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel_button.clicked.connect(self.cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(buttons)

    def cancel(self) -> None:
        self.cancelRequested.emit()

    def set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def update_progress(self, done: int, total: int) -> None:
        maximum = max(1, int(total))
        bounded = max(0, min(int(done), maximum))
        self._progress_bar.setRange(0, maximum)
        self._progress_bar.setValue(bounded)

    def show_cancelling(self) -> None:
        self._cancel_button.setEnabled(False)
        self._status_label.setText("Cancelling import")
        self.setWindowTitle("Cancelling TIFF Import")


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
        self._project_metadata_dirty = False
        self._save_path: Optional[Path] = None
        self._export_path: Optional[Path] = None
        self._settings = settings or APP_SETTINGS
        self._legend_action_group: Optional[QActionGroup] = None
        self._tool_action_group: Optional[QActionGroup] = None
        self._undo_stack = UndoStack(on_changed=self._on_project_state_changed)
        self._tools: Dict[str, Tool] = {
            "navigate": NavigateTool(self),
            "paint": PaintTool(self),
        }
        self._active_tool: Optional[Tool] = None
        self._active_tool_name: Optional[str] = None
        self._file_menu: Optional[QMenu] = None
        self._edit_menu: Optional[QMenu] = None
        self._undo_action: Optional[QAction] = None
        self._redo_action: Optional[QAction] = None
        self._tools_menu: Optional[QMenu] = None
        self._view_menu: Optional[QMenu] = None
        self._current_legend = self._load_current_legend()
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[IcosphereBuildWorker] = None
        self._progress_dialog: Optional[IcosphereProgressDialog] = None
        self._import_thread: Optional[QThread] = None
        self._import_worker: Optional[TiffImportWorker] = None
        self._import_progress_dialog: Optional[TiffImportProgressDialog] = None
        self._pending_import_path: Optional[Path] = None
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[TiffExportWorker] = None
        self._export_progress_dialog: Optional[TiffExportProgressDialog] = None
        self._pending_export_path: Optional[Path] = None
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

        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_project)
        self._file_menu.addAction(new_action)

        self._file_menu.addSeparator()

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

        import_action = QAction("Import", self)
        import_action.triggered.connect(self.import_tiff)
        self._file_menu.addAction(import_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_tiff)
        self._file_menu.addAction(export_action)

        export_as_action = QAction("Export As...", self)
        export_as_action.triggered.connect(self.export_tiff_as)
        self._file_menu.addAction(export_as_action)

        self._file_menu.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        self._file_menu.addAction(settings_action)

        self._edit_menu = self.menuBar().addMenu("Edit")

        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Undo))
        self._undo_action.triggered.connect(self._undo_stack.undo)
        self._edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("Redo", self)
        self._redo_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Redo))
        self._redo_action.triggered.connect(self._undo_stack.redo)
        self._edit_menu.addAction(self._redo_action)

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
        self._legend_menu.setToolTipsVisible(True)
        self._legend_menu.aboutToShow.connect(self._populate_legend_menu)
        self._sync_edit_actions()

    def _set_active_tool(self, tool_name: str) -> None:
        next_tool = self._tools.get(tool_name)
        if next_tool is None:
            raise ValueError(f"Unknown tool selection: {tool_name}")

        if self._active_tool is not None and self._active_tool is not next_tool:
            self._active_tool.dispose_options_dialog()
        self._active_tool = next_tool
        self._active_tool_name = tool_name
        self._active_tool.show_options_dialog()
        self._sync_tool_mode_to_viewer()

    def _sync_tool_mode_to_viewer(self) -> None:
        gl_widget = getattr(self.viewer, "gl_widget", None)
        if gl_widget is None:
            return
        gl_widget.set_tool_mode(self._active_tool_name or "navigate")
        paint_tool = self._tools.get("paint")
        if isinstance(paint_tool, PaintTool):
            gl_widget.set_paint_brush(paint_tool.radius_km(), paint_tool.falloff_percent())
            gl_widget.set_brush_command_factory(self._create_brush_command)
        gl_widget.set_undo_stack(self._undo_stack)

    def _create_brush_command(self, payload: Dict[str, object]) -> BrushPaintCommand:
        paint_tool = self._tools.get("paint")
        if not isinstance(paint_tool, PaintTool):
            raise RuntimeError("Paint tool is not available.")
        mesh_grid = self.viewer.gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Paint tool requires a mesh-backed globe.")

        return BrushPaintCommand(
            target_lat_deg=float(payload["target_lat_deg"]),
            target_lon_deg=float(payload["target_lon_deg"]),
            radius_km=float(payload["radius_km"]),
            falloff_percent=float(payload["falloff_percent"]),
            paint_value=paint_tool.value(),
            paint_mode=paint_tool.mode(),
            mesh_grid=mesh_grid,
            apply_vertex_values=self.viewer.gl_widget.apply_mesh_vertex_values,
        )

    def _sync_edit_actions(self) -> None:
        if self._undo_action is not None:
            self._undo_action.setEnabled(self._undo_stack.can_undo())
        if self._redo_action is not None:
            self._redo_action.setEnabled(self._undo_stack.can_redo())

    def _on_project_state_changed(self) -> None:
        self._sync_edit_actions()
        self.setWindowModified(self._has_unsaved_changes())

    def _has_unsaved_changes(self) -> bool:
        return not self._undo_stack.is_clean() or self._project_metadata_dirty

    def _set_project_window_title(self, title: str) -> None:
        self.setWindowTitle(f"{title}[*]")

    def _mark_project_clean(self) -> None:
        self._project_metadata_dirty = False
        self._undo_stack.set_clean()

    def _mark_project_metadata_dirty(self) -> None:
        self._project_metadata_dirty = True
        self._on_project_state_changed()

    def _confirm_discard_or_save_changes(self, action_description: str) -> bool:
        if not self._has_unsaved_changes():
            return True

        choice = QMessageBox.warning(
            self,
            "Unsaved project changes",
            f"Do you want to save your changes before {action_description}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Discard:
            return True
        return self.save_heightfield()

    def _legend_name(self, legend_source: Union[str, Path]) -> str:
        return Path(str(legend_source).removeprefix(":/")).stem

    def _available_legend_sources(self) -> list[str]:
        resource_dir = QDir(":/legends")
        resource_legends = resource_dir.entryList(["*.txt"], QDir.Filter.Files, QDir.SortFlag.Name)
        combined: list[str] = [f":/legends/{legend}" for legend in resource_legends]
        if LEGENDS_DIR.is_dir():
            combined.extend(str(path) for path in sorted(LEGENDS_DIR.glob("*.txt")))

        legends_by_name: dict[str, str] = {}
        for legend in combined:
            legends_by_name.setdefault(self._legend_name(legend), legend)

        return [legends_by_name[name] for name in sorted(legends_by_name)]

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

    def _legend_icon(self, legend_source: str) -> QIcon:
        cmap = load_topo_cmap(legend_source, name=self._legend_name(legend_source))
        samples = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        rgba = np.asarray(cmap(samples), dtype=np.float32)
        if rgba.ndim == 1:
            rgba = np.tile(rgba, (256, 1))

        pixmap = QPixmap(320, 48)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            gradient = QLinearGradient(0, 0, 160, 0)
            for idx in range(rgba.shape[0]):
                color = QColor.fromRgbF(
                    float(rgba[idx, 0]),
                    float(rgba[idx, 1]),
                    float(rgba[idx, 2]),
                    float(rgba[idx, 3]),
                )
                gradient.setColorAt(float(idx) / max(1, rgba.shape[0] - 1), color)
            painter.fillRect(0, 0, 160, 16, gradient)
            painter.setPen(QColor(0, 0, 0, 80))
            painter.drawRect(0, 0, 159, 15)
        finally:
            painter.end()
        return QIcon(pixmap)

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
            action.setIcon(self._legend_icon(legend_source))
            action.setCheckable(True)
            action.setChecked(legend_source == current_legend)
            action.triggered.connect(lambda checked=False, source=legend_source: self.set_legend(source))
            self._legend_action_group.addAction(action)

    def set_legend(self, legend_source: Union[str, Path]) -> None:
        legend_value = str(legend_source)
        if legend_value not in self._available_legend_sources():
            raise FileNotFoundError(f"Legend not found: {legend_value}")
        if legend_value == self._current_legend:
            return

        self._current_legend = legend_value
        self._settings.setValue("legend/current", legend_value)
        self.viewer.set_colormap(load_topo_cmap(legend_value, name=self._legend_name(legend_value)))
        self._mark_project_metadata_dirty()

    def new_project(self) -> None:
        if not self._confirm_discard_or_save_changes("starting a new project"):
            return
        self._set_heightfield(None)

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

    def _start_tiff_import(self, path: Path) -> None:
        if self._import_thread is not None:
            raise RuntimeError("A TIFF import is already in progress.")
        if self._load_thread is not None:
            raise RuntimeError("Wait for the current heightfield load to finish before importing.")
        if self._export_thread is not None:
            raise RuntimeError("Wait for the current TIFF export to finish before importing.")

        self._pending_import_path = Path(path)
        self._import_progress_dialog = TiffImportProgressDialog(self)
        self._import_thread = QThread(self)
        self._import_worker = TiffImportWorker(
            self._pending_import_path,
            min_level=self._mesh_min_level,
            max_level=self._mesh_max_level,
            threshold=self._mesh_threshold,
        )
        self._import_worker.moveToThread(self._import_thread)

        self._import_thread.started.connect(self._import_worker.run)
        self._import_worker.progressMessageChanged.connect(self._import_progress_dialog.set_status)
        self._import_worker.progressChanged.connect(self._import_progress_dialog.update_progress)
        self._import_worker.completed.connect(self._on_tiff_import_completed)
        self._import_worker.failed.connect(self._on_tiff_import_failed)
        self._import_worker.cancelled.connect(self._on_tiff_import_cancelled)
        self._import_worker.completed.connect(self._import_thread.quit)
        self._import_worker.failed.connect(self._import_thread.quit)
        self._import_worker.cancelled.connect(self._import_thread.quit)
        self._import_thread.finished.connect(self._cleanup_tiff_import)
        self._import_progress_dialog.cancelRequested.connect(self._import_worker.cancel)
        self._import_progress_dialog.cancelRequested.connect(self._import_progress_dialog.show_cancelling)

        self._import_progress_dialog.show()
        self._import_thread.start()

    @Slot(object)
    def _on_tiff_import_completed(self, payload: object) -> None:
        if self._pending_import_path is None:
            raise RuntimeError("TIFF import completed without a pending import path.")
        if (
            not isinstance(payload, tuple)
            or len(payload) != 2
        ):
            raise RuntimeError(f"TIFF import completed with unexpected payload: {payload!r}")

        raster, grid = payload
        raster_data = np.asarray(raster, dtype=np.float32)
        if not isinstance(grid, IcosphereGrid):
            raise RuntimeError(f"TIFF import completed with unexpected grid payload: {grid!r}")
        self._set_heightfield(self._pending_import_path, elevation=raster_data, mesh_grid=grid)
        self._close_import_progress_dialog()

    @Slot(object)
    def _on_tiff_import_failed(self, exc: object) -> None:
        self._close_import_progress_dialog()
        if not isinstance(exc, Exception):
            raise RuntimeError(f"TIFF import failed with unexpected error payload: {exc!r}")
        QMessageBox.critical(self, "Failed to import TIFF", str(exc))

    @Slot()
    def _on_tiff_import_cancelled(self) -> None:
        self._close_import_progress_dialog()

    def _close_import_progress_dialog(self) -> None:
        if self._import_progress_dialog is None:
            return
        self._import_progress_dialog.close()
        self._import_progress_dialog.deleteLater()
        self._import_progress_dialog = None

    @Slot()
    def _cleanup_tiff_import(self) -> None:
        if self._import_worker is not None:
            self._import_worker.deleteLater()
            self._import_worker = None
        if self._import_thread is not None:
            self._import_thread.deleteLater()
            self._import_thread = None
        self._pending_import_path = None

    def _start_tiff_export(self, path: Path) -> None:
        if self._export_thread is not None:
            raise RuntimeError("A TIFF export is already in progress.")
        if self._import_thread is not None:
            raise RuntimeError("Wait for the current TIFF import to finish before exporting.")
        if self._load_thread is not None:
            raise RuntimeError("Wait for the current heightfield load to finish before exporting.")

        mesh_grid = self.viewer.gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("TIFF export requires an icosphere grid.")

        self._pending_export_path = Path(path)
        self._export_progress_dialog = TiffExportProgressDialog(self)
        self._export_thread = QThread(self)
        self._export_worker = TiffExportWorker(mesh_grid, self._pending_export_path)
        self._export_worker.moveToThread(self._export_thread)

        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.progressMessageChanged.connect(self._export_progress_dialog.set_status)
        self._export_worker.progressChanged.connect(self._export_progress_dialog.update_progress)
        self._export_worker.completed.connect(self._on_tiff_export_completed)
        self._export_worker.failed.connect(self._on_tiff_export_failed)
        self._export_worker.cancelled.connect(self._on_tiff_export_cancelled)
        self._export_worker.completed.connect(self._export_thread.quit)
        self._export_worker.failed.connect(self._export_thread.quit)
        self._export_worker.cancelled.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._cleanup_tiff_export)
        self._export_progress_dialog.cancelRequested.connect(self._export_worker.cancel)
        self._export_progress_dialog.cancelRequested.connect(self._export_progress_dialog.show_cancelling)

        self._export_progress_dialog.show()
        self._export_thread.start()

    @Slot()
    def _on_tiff_export_completed(self) -> None:
        self._close_export_progress_dialog()

    @Slot(object)
    def _on_tiff_export_failed(self, exc: object) -> None:
        self._close_export_progress_dialog()
        if not isinstance(exc, Exception):
            raise RuntimeError(f"TIFF export failed with unexpected error payload: {exc!r}")
        QMessageBox.critical(self, "Failed to export TIFF", str(exc))

    @Slot()
    def _on_tiff_export_cancelled(self) -> None:
        self._close_export_progress_dialog()

    def _close_export_progress_dialog(self) -> None:
        if self._export_progress_dialog is None:
            return
        self._export_progress_dialog.close()
        self._export_progress_dialog.deleteLater()
        self._export_progress_dialog = None

    @Slot()
    def _cleanup_tiff_export(self) -> None:
        if self._export_worker is not None:
            self._export_worker.deleteLater()
            self._export_worker = None
        if self._export_thread is not None:
            self._export_thread.deleteLater()
            self._export_thread = None
        self._pending_export_path = None

    def _set_heightfield(
        self,
        heightfield: Optional[Path],
        *,
        elevation: Optional[np.ndarray] = None,
        mesh_grid: Optional[IcosphereGrid] = None,
    ) -> None:
        self._undo_stack.clear()
        existing_settings = None
        if isinstance(self.centralWidget(), GlobeViewer):
            existing_settings = self._current_grid_settings()

        if heightfield is None:
            elevation_data = empty_globe_elevation() if elevation is None else np.asarray(elevation, dtype=np.float32)
            view_title = "Sea level"
            mesh = mesh_grid if mesh_grid is not None else generic_icosphere_mesh(self._mesh_max_level)
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

            self._set_project_window_title(view_title)
            self._heightfield = None
            self._save_path = None
            self._export_path = None
            self._sync_tool_mode_to_viewer()
            self._mark_project_clean()
            return

        heightfield = Path(heightfield).expanduser().resolve()
        if not heightfield.is_file():
            raise FileNotFoundError(f"Heightfield not found: {heightfield}")

        if Project.is_project_path(heightfield):
            project = Project.load(heightfield)
            legend_source = project.metadata.get("legend_source")
            if isinstance(legend_source, str) and legend_source in self._available_legend_sources():
                self._current_legend = legend_source
            elevation_data = project.raster
            mesh_grid = project.grid
        elif elevation is not None:
            elevation_data = np.asarray(elevation, dtype=np.float32)
        elif heightfield.suffix.lower() in (".npy", ".npz"):
            elevation_data = self._load_numpy_heightfield(heightfield)
        else:
            elevation_data = load_elevation(heightfield)
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

        self._set_project_window_title(heightfield.stem)
        self._heightfield = heightfield
        self._save_path = heightfield if Project.is_project_path(heightfield) else None
        self._export_path = None
        self._sync_tool_mode_to_viewer()
        self._mark_project_clean()

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
        if not self._confirm_discard_or_save_changes("closing the application"):
            event.ignore()
            return
        for tool in self._tools.values():
            tool.dispose_options_dialog()
        if self._import_worker is not None:
            self._import_worker.cancel()
        if self._import_thread is not None:
            self._import_thread.quit()
            self._import_thread.wait()
        if self._export_worker is not None:
            self._export_worker.cancel()
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait()
        if self._load_worker is not None:
            self._load_worker.cancel()
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()
        self._save_grid_settings()
        self._settings.sync()
        super().closeEvent(event)

    def _create_open_heightfield_dialog(self) -> QFileDialog:
        starting_dir = self._heightfield.parent if self._heightfield is not None else Path.cwd()
        dialog = QFileDialog(self, "Open Heightfield", str(starting_dir))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(
            f"Oswald Globe Projects (*{Project.FILE_SUFFIX});;"
            "NumPy Arrays (*.npy *.npz);;"
            "All Files (*)"
        )
        return dialog

    @staticmethod
    def _load_numpy_heightfield(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                key = "elevation" if "elevation" in archive.files else archive.files[0]
                return np.asarray(archive[key], dtype=np.float32)
        raise ValueError(f"Unsupported NumPy heightfield format: {path.suffix}")

    def open_heightfield(self) -> None:
        dialog = self._create_open_heightfield_dialog()
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return

        heightfield = Path(selected_files[0])
        if Project.is_project_path(heightfield):
            try:
                self._set_heightfield(heightfield)
            except Exception as exc:
                QMessageBox.critical(self, "Failed to open project", str(exc))
            return
        progress_dialog = IcosphereProgressDialog(self)
        self._progress_dialog = progress_dialog
        progress_dialog.start_stage("loading-heightfield")
        progress_dialog.show()
        try:
            elevation = self._load_numpy_heightfield(heightfield)
        except Exception as exc:
            progress_dialog.close()
            progress_dialog.deleteLater()
            self._progress_dialog = None
            QMessageBox.critical(self, "Failed to open heightfield", str(exc))
            return
        progress_dialog.finish_stage("loading-heightfield")
        self._start_heightfield_load(heightfield, elevation)

    def _create_import_tiff_dialog(self) -> QFileDialog:
        starting_dir = self._heightfield.parent if self._heightfield is not None else Path.cwd()
        dialog = QFileDialog(self, "Import TIFF", str(starting_dir))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("TIFF Images (*.tif *.tiff);;All Files (*)")
        return dialog

    def import_tiff(self) -> None:
        dialog = self._create_import_tiff_dialog()
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return

        heightfield = Path(selected_files[0])
        if heightfield.suffix.lower() not in (".tif", ".tiff"):
            QMessageBox.critical(self, "Failed to import TIFF", "Select a .tif or .tiff file.")
            return
        try:
            self._start_tiff_import(heightfield)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to import TIFF", str(exc))

    def save_heightfield(self) -> bool:
        if self._save_path is None:
            return self.save_heightfield_as()
        try:
            self._save_heightfield(self._save_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save project", str(exc))
            return False
        self._save_path = self._save_path
        self._mark_project_clean()
        return True

    def save_heightfield_as(self) -> bool:
        default_path = (
            self._heightfield.with_suffix(Project.FILE_SUFFIX)
            if self._heightfield is not None
            else Path(f"sea_level{Project.FILE_SUFFIX}")
        )
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            str(default_path),
            (
                f"Oswald Globe Project (*{Project.FILE_SUFFIX});;"
                "NumPy Array (*.npy);;"
                "Compressed NumPy Array (*.npz)"
            ),
        )
        if not file_name:
            return False
        save_path = Path(file_name)
        if not save_path.suffix:
            save_path = save_path.with_suffix(Project.FILE_SUFFIX)
        elif save_path.suffix.lower() not in (Project.FILE_SUFFIX, ".npy", ".npz"):
            QMessageBox.critical(
                self,
                "Failed to save project",
                f"Unsupported save format: {save_path.suffix}",
            )
            return False
        try:
            self._save_heightfield(save_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save project", str(exc))
            return False
        self._save_path = save_path
        self._mark_project_clean()
        return True

    def _save_heightfield(self, path: Path) -> None:
        path = Path(path)
        gl_widget = self.viewer.gl_widget
        data = gl_widget.current_raster_data() if gl_widget.mesh_grid is not None else self.viewer.data
        suffix = path.suffix.lower()
        if suffix == Project.FILE_SUFFIX:
            mesh_grid = gl_widget.mesh_grid
            if mesh_grid is None:
                raise RuntimeError("Project saves require an icosphere grid.")
            Project(
                raster=data,
                grid=mesh_grid,
                metadata={"legend_source": self._current_legend},
            ).save(path)
        elif suffix == ".npy":
            np.save(path, data)
        elif suffix == ".npz":
            np.savez_compressed(path, elevation=data)
        else:
            raise ValueError(f"Unsupported save format: {path.suffix}")

    def export_tiff(self) -> None:
        if self._export_path is None:
            self.export_tiff_as()
            return
        try:
            self._start_tiff_export(self._export_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to export TIFF", str(exc))

    def export_tiff_as(self) -> None:
        default_path = (
            self._heightfield.with_suffix(".tif")
            if self._heightfield is not None
            else Path("sea_level.tif")
        )
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export TIFF",
            str(default_path),
            "TIFF Image (*.tif *.tiff)",
        )
        if file_name:
            export_path = Path(file_name)
            if export_path.suffix.lower() not in (".tif", ".tiff"):
                if export_path.suffix:
                    QMessageBox.critical(self, "Failed to export TIFF", "Choose a .tif or .tiff file.")
                    return
                export_path = export_path.with_suffix(".tif")
            self._export_path = export_path
            try:
                self._start_tiff_export(self._export_path)
            except Exception as exc:
                QMessageBox.critical(self, "Failed to export TIFF", str(exc))

    def _export_tiff(self, path: Path) -> None:
        mesh_grid = self.viewer.gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("TIFF export requires an icosphere grid.")
        raster = mesh_grid.to_equirectangular(height=2048, width=4096)
        _write_tiff_export(Path(path), raster)


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
    parser.add_argument("--mesh-max-level", type=int, default=6, help="Maximum icosphere subdivision level.")
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
