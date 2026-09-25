import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QSettings, Qt, QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QFileDialog, QSlider

from oswald_globe.app import IcosphereProgressDialog, create_window, load_elevation
from oswald_globe.icosphere import IcosphereGrid
from oswald_globe.undo_stack import BrushPaintCommand, UndoStack


def test_load_elevation(tmp_path: Path):
    tif_path = tmp_path / "test.tif"
    arr = np.ones((10, 20), dtype=np.float32) * 0.5
    img = Image.fromarray(arr)
    img.save(tif_path)

    elev = load_elevation(tif_path)
    assert elev.shape == (10, 20)
    # 0.5 * 4096.0 - 1024.0 == 1024.0
    np.testing.assert_allclose(elev, 1024.0, atol=1e-3)


def test_create_window_defaults_to_empty_sea_level_globe(qapp):
    window = create_window()
    try:
        assert window._heightfield is None
        assert window.windowTitle() == "Sea level"
        assert window.viewer.data.shape == (2, 2)
        np.testing.assert_allclose(window.viewer.data, 0.0)
        assert window.viewer.mesh_grid is not None
        assert len(window.viewer.mesh_grid.vertices) > 0
    finally:
        window.close()


def test_resources_module_importable():
    from oswald_globe import resources_rc

    assert hasattr(resources_rc, "qInitResources")


def test_app_help_when_run_as_script():
    project_root = Path(__file__).resolve().parent.parent
    app_path = project_root / "oswald_globe" / "app.py"
    result = subprocess.run(
        [sys.executable, str(app_path), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Interactive 3D heightfield desktop globe." in result.stdout


def test_create_window(qapp, tmp_path: Path):
    tif_path = tmp_path / "globe.tif"
    arr = np.zeros((32, 64), dtype=np.float32)
    img = Image.fromarray(arr)
    img.save(tif_path)

    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    window = create_window(
        heightfield=tif_path,
        mesh_min_level=1,
        mesh_max_level=2,
        mesh_threshold=200.0,
        settings=settings,
    )
    assert window is not None
    assert window.windowTitle() == "globe"
    assert window.centralWidget() is not None
    assert [action.text() for action in window.menuBar().actions()] == ["File", "Edit", "Tools", "View"]
    assert window._file_menu is not None
    assert [action.text() for action in window._file_menu.actions() if action.text()] == [
        "Open",
        "Save",
        "Save As...",
        "Settings",
    ]
    assert window._edit_menu is not None
    assert [action.text() for action in window._edit_menu.actions()] == ["Undo", "Redo"]
    assert [action.isEnabled() for action in window._edit_menu.actions()] == [False, False]
    assert window._tools_menu is not None
    assert [action.text() for action in window._tools_menu.actions()] == ["Navigate", "Paint"]
    assert [action.text() for action in window._tools_menu.actions() if action.isChecked()] == ["Navigate"]

    assert window._legend_menu is not None
    legend_menu = window._legend_menu
    window._populate_legend_menu()
    assert [action.text() for action in legend_menu.actions()] == ["grayscale", "topography"]
    assert [action.text() for action in legend_menu.actions() if action.isChecked()] == ["topography"]

    window.set_legend(":/legends/grayscale.txt")
    assert window.viewer.cmap.name == "grayscale"
    window._populate_legend_menu()
    assert [action.text() for action in legend_menu.actions() if action.isChecked()] == ["grayscale"]

    window.apply_grid_settings(
        {
            "icosphere": {"visible": False, "color": QColor("#ff0000"), "opacity": 0.25},
            "graticule": {"visible": True, "color": QColor("#00ff00"), "opacity": 0.75},
        }
    )
    viewer = window.centralWidget()
    assert viewer.gl_widget.show_wireframe is False
    assert viewer.gl_widget.btn_mesh.isChecked() is False
    assert viewer.gl_widget.mesh_color == (1.0, 0.0, 0.0)
    assert viewer.gl_widget.mesh_opacity == 0.25
    assert viewer.gl_widget.is_graticule is True
    assert viewer.gl_widget.btn_grid.isChecked() is True
    assert viewer.gl_widget.graticule_color == (0.0, 1.0, 0.0)
    assert viewer.gl_widget.graticule_opacity == 0.75

    window.close()

    restored_window = create_window(
        heightfield=tif_path,
        mesh_min_level=1,
        mesh_max_level=2,
        mesh_threshold=200.0,
        settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
    )
    restored_viewer = restored_window.centralWidget()
    restored_window._populate_legend_menu()
    assert restored_window._legend_menu is not None
    restored_legend_menu = restored_window._legend_menu
    assert restored_viewer.cmap.name == "grayscale"
    assert [action.text() for action in restored_legend_menu.actions() if action.isChecked()] == ["grayscale"]
    assert restored_viewer.gl_widget.show_wireframe is False
    assert restored_viewer.gl_widget.btn_mesh.isChecked() is False
    assert restored_viewer.gl_widget.mesh_color == (1.0, 0.0, 0.0)
    assert restored_viewer.gl_widget.mesh_opacity == 0.25
    assert restored_viewer.gl_widget.graticule_color == (0.0, 1.0, 0.0)
    assert restored_viewer.gl_widget.graticule_opacity == 0.75
    restored_window.close()


def test_paint_tool_selection_shows_options_and_navigate_disposes(qapp):
    window = create_window()
    try:
        assert window._tools_menu is not None
        tools_menu = window._tools_menu
        navigate_action, paint_action = tools_menu.actions()
        paint_tool = window._tools["paint"]

        assert navigate_action.isChecked() is True
        assert paint_action.isChecked() is False
        assert paint_tool.options_dialog is None

        paint_action.trigger()
        assert paint_action.isChecked() is True
        assert navigate_action.isChecked() is False
        assert paint_tool.options_dialog is not None
        assert paint_tool.options_dialog.windowTitle() == "Paint Tool Options"
        assert bool(paint_tool.options_dialog.windowFlags() & Qt.WindowType.Tool) is True
        assert bool(paint_tool.options_dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) is True

        sliders = paint_tool.options_dialog.findChildren(QSlider)
        ranges = sorted((slider.minimum(), slider.maximum()) for slider in sliders)
        assert ranges == [(0, 100), (0, 100), (0, 1000)]
        mode_combo = paint_tool.options_dialog.findChild(QComboBox)
        assert mode_combo is not None
        assert [mode_combo.itemText(i) for i in range(mode_combo.count())] == [
            "Replace",
            "Add",
            "Subtract",
            "Minimum",
            "Maximum",
            "Multiply",
            "Average",
        ]
        assert paint_tool.value() == 50
        assert paint_tool.mode() == "Replace"
        assert paint_tool.radius_km() == 100
        assert paint_tool.falloff_percent() == 50
        paint_tool.options_dialog.value_slider.setValue(73)
        mode_combo.setCurrentText("Multiply")
        paint_tool.options_dialog.radius_slider.setValue(444)
        paint_tool.options_dialog.falloff_slider.setValue(11)

        navigate_action.trigger()
        assert navigate_action.isChecked() is True
        assert paint_action.isChecked() is False
        assert paint_tool.options_dialog is None
        assert paint_tool.value() == 73
        assert paint_tool.mode() == "Multiply"
        assert paint_tool.radius_km() == 444
        assert paint_tool.falloff_percent() == 11

        paint_action.trigger()
        assert paint_tool.options_dialog is not None
        assert paint_tool.options_dialog.value_slider.value() == 73
        assert paint_tool.options_dialog.mode_combo.currentText() == "Multiply"
        assert paint_tool.options_dialog.radius_slider.value() == 444
        assert paint_tool.options_dialog.falloff_slider.value() == 11
    finally:
        window.close()


def test_paint_brush_release_pushes_undo_command(qapp, monkeypatch: pytest.MonkeyPatch, capsys):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")

        stack = UndoStack()
        widget.set_undo_stack(stack)

        payloads = []

        def factory(payload):
            payloads.append(payload)
            return BrushPaintCommand(
                target_lat_deg=float(payload["target_lat_deg"]),
                target_lon_deg=float(payload["target_lon_deg"]),
                radius_km=float(payload["radius_km"]),
                falloff_percent=float(payload["falloff_percent"]),
                paint_value=99,
                paint_mode="Replace",
            )

        widget.set_brush_command_factory(factory)
        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {"target_lat": 0.5, "target_lon": -1.0, "value": 0.0, "lat_deg": 0.0, "lon_deg": 0.0},
        )

        class PaintEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.LeftButton

            def buttons(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

        widget.mouseMoveEvent(PaintEvent(12.0, 34.0))
        widget.mouseReleaseEvent(PaintEvent(12.0, 34.0))

        assert payloads == [
            {
                "target_lat_deg": pytest.approx(28.64788975654116),
                "target_lon_deg": pytest.approx(-57.29577951308232),
                "radius_km": pytest.approx(100.0),
                "falloff_percent": pytest.approx(50.0),
            },
            {
                "target_lat_deg": pytest.approx(28.64788975654116),
                "target_lon_deg": pytest.approx(-57.29577951308232),
                "radius_km": pytest.approx(100.0),
                "falloff_percent": pytest.approx(50.0),
            }
        ]
        assert stack.can_undo() is True
        assert stack.can_redo() is False
        out = capsys.readouterr().out
        assert out.count("[brush] apply paint stroke placeholder") == 2

        stack.undo()
        out = capsys.readouterr().out
        assert "[brush] undo paint stroke placeholder" in out

        stack.redo()
        out = capsys.readouterr().out
        assert "[brush] apply paint stroke placeholder" in out
    finally:
        window.close()


def test_paint_drag_updates_brush_hover_position(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")

        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(
            lambda payload: BrushPaintCommand(
                target_lat_deg=float(payload["target_lat_deg"]),
                target_lon_deg=float(payload["target_lon_deg"]),
                radius_km=float(payload["radius_km"]),
                falloff_percent=float(payload["falloff_percent"]),
                paint_value=99,
                paint_mode="Replace",
            )
        )

        updates = []
        monkeypatch.setattr(widget, "update", lambda: updates.append(True))
        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {"target_lat": 0.5, "target_lon": -1.0, "value": 0.0, "lat_deg": 0.0, "lon_deg": 0.0},
        )

        class PaintEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def buttons(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

        widget.mouseMoveEvent(PaintEvent(12.0, 34.0))

        assert widget._hover_target_lat == pytest.approx(0.5)
        assert widget._hover_target_lon == pytest.approx(-1.0)
        assert updates
    finally:
        window.close()


def test_middle_drag_pans_globe_in_paint_mode(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")

        widget.set_undo_stack(UndoStack())
        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {
                "target_lat": 0.5,
                "target_lon": -1.0,
                "value": 0.0,
                "lat_deg": 0.0,
                "lon_deg": 0.0,
            },
        )

        class MiddleEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.MiddleButton

            def buttons(self):
                return Qt.MouseButton.MiddleButton

            def position(self):
                return self._pos

        class HoverEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def buttons(self):
                return Qt.MouseButton.NoButton

            def position(self):
                return self._pos

        start_lon = widget.center_lon
        start_lat = widget.center_lat

        widget.mouseMoveEvent(HoverEvent(12.0, 34.0))
        assert widget.cursor().shape() == Qt.CursorShape.BlankCursor

        widget.mousePressEvent(MiddleEvent(12.0, 34.0))
        assert widget.cursor().shape() == Qt.CursorShape.ClosedHandCursor
        widget.mouseMoveEvent(MiddleEvent(22.0, 34.0))
        assert widget.cursor().shape() == Qt.CursorShape.ClosedHandCursor
        widget.mouseReleaseEvent(MiddleEvent(22.0, 34.0))
        assert widget.cursor().shape() == Qt.CursorShape.BlankCursor

        assert widget.center_lon < start_lon
        assert widget.center_lat == pytest.approx(start_lat)
        assert widget.is_dragging is False
    finally:
        window.close()


def test_open_heightfield_dialog_configuration(qapp, tmp_path: Path):
    tif_path = tmp_path / "globe.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(tif_path)
    window = create_window(heightfield=tif_path, mesh_min_level=1, mesh_max_level=2, mesh_threshold=200.0)

    dialog = window._create_open_heightfield_dialog()

    assert dialog.acceptMode() == dialog.AcceptMode.AcceptOpen
    assert dialog.fileMode() == dialog.FileMode.ExistingFile
    assert dialog.nameFilters()[0].startswith("Image Files (")
    assert "*.tif" in dialog.nameFilters()[0]
    assert "*.png" in dialog.nameFilters()[0]

    window.close()


def test_icosphere_progress_dialog_lists_all_stages(qapp):
    dialog = IcosphereProgressDialog()

    assert tuple(phase for phase, _ in dialog._STAGES) == (
        "loading-heightfield",
        "creating-faces",
        "balancing-faces",
        "populating-faces",
        "displaying-globe",
    )
    assert set(dialog._stage_bars) == {phase for phase, _ in dialog._STAGES}
    assert set(dialog._stage_labels) == {phase for phase, _ in dialog._STAGES}

    dialog.deleteLater()


def test_open_heightfield_cancel_is_noop(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_path = tmp_path / "original.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(original_path)
    window = create_window(heightfield=original_path, mesh_min_level=1, mesh_max_level=2, mesh_threshold=200.0)
    original_viewer = window.viewer

    class CancelDialog:
        def exec(self):
            return QFileDialog.DialogCode.Rejected

    monkeypatch.setattr(window, "_create_open_heightfield_dialog", lambda: CancelDialog())

    window.open_heightfield()

    assert window._heightfield == original_path.resolve()
    assert window.viewer is original_viewer
    assert window.windowTitle() == "original"

    window.close()


def test_open_heightfield_loads_selected_image(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_path = tmp_path / "original.tif"
    replacement_path = tmp_path / "replacement.png"

    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(original_path)
    replacement_data = np.linspace(0, 255, 64 * 128, dtype=np.uint8).reshape(64, 128)
    Image.fromarray(replacement_data, mode="L").save(replacement_path)

    window = create_window(heightfield=original_path, mesh_min_level=1, mesh_max_level=3, mesh_threshold=50.0)
    original_viewer = window.viewer
    original_mesh = original_viewer.mesh_grid
    built_mesh = IcosphereGrid.from_equirectangular(
        load_elevation(replacement_path),
        min_level=1,
        max_level=3,
        threshold=50.0,
    )

    class AcceptDialog:
        def exec(self):
            return QFileDialog.DialogCode.Accepted

        def selectedFiles(self):
            return [str(replacement_path)]

    monkeypatch.setattr(window, "_create_open_heightfield_dialog", lambda: AcceptDialog())
    monkeypatch.setattr(window, "_start_heightfield_load", lambda heightfield, elevation: window._set_heightfield(heightfield, elevation=elevation, mesh_grid=built_mesh))

    window.open_heightfield()

    assert window._heightfield == replacement_path.resolve()
    assert window.windowTitle() == "replacement"
    assert window.viewer is not original_viewer
    assert window.viewer.mesh_grid is not None
    assert window.viewer.mesh_grid is not original_mesh
    assert window.viewer.mesh_grid is built_mesh
    assert window.viewer.data.shape == replacement_data.shape

    window.close()
