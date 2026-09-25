import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QSettings, Qt, QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QFileDialog, QSlider, QSpinBox, QToolTip

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
        np.testing.assert_allclose(window.viewer.mesh_grid.values, 0.0)
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
    assert window._legend_menu.toolTipsVisible() is True
    legend_menu = window._legend_menu
    window._populate_legend_menu()
    assert [action.text() for action in legend_menu.actions()] == ["grayscale", "topography"]
    assert all(action.icon().isNull() for action in legend_menu.actions())
    assert all(action.defaultWidget() is not None for action in legend_menu.actions())
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
        assert ranges == [(-32767, 32767), (0, 100), (0, 1000)]
        spin_boxes = paint_tool.options_dialog.findChildren(QSpinBox)
        spin_ranges = sorted((spin.minimum(), spin.maximum()) for spin in spin_boxes)
        assert spin_ranges == [(-32767, 32767), (0, 100), (0, 1000)]
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
        assert paint_tool.value() == 0
        assert paint_tool.mode() == "Replace"
        assert paint_tool.radius_km() == 100
        assert paint_tool.falloff_percent() == 50
        assert paint_tool.options_dialog.value_spin.value() == 0
        assert paint_tool.options_dialog.radius_spin.value() == 100
        assert paint_tool.options_dialog.falloff_spin.value() == 50
        assert paint_tool.options_dialog.value_spin.singleStep() == 256
        assert paint_tool.options_dialog.radius_spin.singleStep() == 50
        assert paint_tool.options_dialog.falloff_spin.singleStep() == 5
        paint_tool.options_dialog.value_slider.setValue(73)
        mode_combo.setCurrentText("Multiply")
        paint_tool.options_dialog.radius_slider.setValue(444)
        paint_tool.options_dialog.falloff_slider.setValue(11)
        assert paint_tool.options_dialog.value_spin.value() == 73
        assert paint_tool.options_dialog.radius_spin.value() == 444
        assert paint_tool.options_dialog.falloff_spin.value() == 11

        paint_tool.options_dialog.value_spin.setValue(61)
        paint_tool.options_dialog.radius_spin.stepUp()
        paint_tool.options_dialog.falloff_spin.stepDown()
        assert paint_tool.options_dialog.value_slider.value() == 61
        assert paint_tool.options_dialog.radius_slider.value() == 494
        assert paint_tool.options_dialog.falloff_slider.value() == 6

        paint_tool.options_dialog.value_spin.stepUp()
        assert paint_tool.options_dialog.value_slider.value() == 317

        paint_tool.options_dialog.radius_spin.lineEdit().setText("2000")
        paint_tool.options_dialog.radius_spin.interpretText()
        assert paint_tool.options_dialog.radius_spin.value() == 494
        assert paint_tool.options_dialog.radius_slider.value() == 494

        navigate_action.trigger()
        assert navigate_action.isChecked() is True
        assert paint_action.isChecked() is False
        assert paint_tool.options_dialog is None
        assert paint_tool.value() == 317
        assert paint_tool.mode() == "Multiply"
        assert paint_tool.radius_km() == 494
        assert paint_tool.falloff_percent() == 6

        paint_action.trigger()
        assert paint_tool.options_dialog is not None
        assert paint_tool.options_dialog.value_slider.value() == 317
        assert paint_tool.options_dialog.mode_combo.currentText() == "Multiply"
        assert paint_tool.options_dialog.radius_slider.value() == 494
        assert paint_tool.options_dialog.falloff_slider.value() == 6
        assert paint_tool.options_dialog.value_spin.value() == 317
        assert paint_tool.options_dialog.radius_spin.value() == 494
        assert paint_tool.options_dialog.falloff_spin.value() == 6
    finally:
        window.close()


@pytest.mark.parametrize(
    ("paint_mode", "paint_value", "expected_inner"),
    [
        ("Replace", 80, lambda original: np.full(original.shape, 80.0, dtype=np.float64)),
        ("Add", 8, lambda original: original + 8.0),
        ("Subtract", 8, lambda original: original - 8.0),
        ("Minimum", 35, lambda original: np.minimum(original, 35.0)),
        ("Maximum", 35, lambda original: np.maximum(original, 35.0)),
        ("Multiply", 2, lambda original: original * 2.0),
        ("Average", 80, lambda original: (original + 80.0) / 2.0),
    ],
)
def test_brush_command_applies_selected_mode_and_tapers_outer_ring(paint_mode, paint_value, expected_inner):
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    original = np.linspace(0.0, 100.0, grid.vertex_count(), dtype=np.float64)
    grid.set_layer("elevation", original.copy())

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0
    radius_km = 8000.0
    falloff_percent = 50.0

    command = BrushPaintCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=radius_km,
        falloff_percent=falloff_percent,
        paint_value=paint_value,
        paint_mode=paint_mode,
        mesh_grid=grid,
        apply_vertex_values=lambda indices, values: grid.get_layer("elevation").values.__setitem__(indices, values),
    )

    center = grid.vertices[target_index]
    distances = np.arccos(np.clip(grid.vertices @ center, -1.0, 1.0))
    outer_radius = np.deg2rad(min(89.0, radius_km / 111.0))
    inner_radius = outer_radius * (falloff_percent / 100.0)
    inner_mask = distances <= inner_radius + 1e-9
    transition_mask = (distances > inner_radius + 1e-9) & (distances <= outer_radius + 1e-9)

    assert np.any(inner_mask)
    assert np.any(transition_mask)

    command.redo()

    expected_operated = expected_inner(original)
    np.testing.assert_allclose(grid.get_layer("elevation").values[inner_mask], expected_operated[inner_mask])
    transition_blend = (distances[transition_mask] - inner_radius) / (outer_radius - inner_radius)
    expected_transition = (
        (1.0 - transition_blend) * expected_operated[transition_mask]
        + transition_blend * original[transition_mask]
    )
    np.testing.assert_allclose(grid.get_layer("elevation").values[transition_mask], expected_transition)

    command.undo()
    np.testing.assert_allclose(grid.get_layer("elevation").values, original)


def test_paint_brush_release_pushes_undo_command(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")

        stack = UndoStack()
        widget.set_undo_stack(stack)

        payloads = []

        def factory(payload):
            payloads.append(payload)
            return window._create_brush_command(payload)

        widget.set_brush_command_factory(factory)
        raster_sync_calls = []
        monkeypatch.setattr(
            widget.mesh_grid,
            "to_equirectangular",
            lambda height, width, k=8: raster_sync_calls.append((height, width, k)) or np.zeros((height, width), dtype=np.float32),
        )
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
        assert raster_sync_calls == []

        stack.undo()
        assert stack.can_redo() is True

        stack.redo()
        assert raster_sync_calls == []
    finally:
        window.close()


def test_paint_drag_updates_brush_hover_position(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")

        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_brush_command)

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


def test_current_raster_data_syncs_mesh_edits_on_demand(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        original_shape = widget.raw_data.shape
        raster_sync_calls = []
        expected = np.full(original_shape, 7.0, dtype=np.float32)
        monkeypatch.setattr(
            widget.mesh_grid,
            "to_equirectangular",
            lambda height, width, k=8: raster_sync_calls.append((height, width, k)) or expected.copy(),
        )

        widget.apply_mesh_vertex_values(np.array([0], dtype=np.intp), np.array([123.0], dtype=np.float64))
        assert raster_sync_calls == []

        current = widget.current_raster_data()
        assert raster_sync_calls == [(original_shape[0], original_shape[1], 8)]
        np.testing.assert_allclose(current, expected)
        np.testing.assert_allclose(window.viewer.data, expected)
    finally:
        window.close()


def test_tooltip_uses_mesh_sample_value(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        shown = {}

        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {
                "target_lat": 0.5,
                "target_lon": -1.0,
                "value": 0.0,
                "lat_deg": 12.0,
                "lon_deg": 34.0,
            },
        )
        monkeypatch.setattr(widget.mesh_grid, "sample", lambda lat, lon, k=8: np.array([123.45], dtype=np.float64))
        monkeypatch.setattr(QToolTip, "showText", lambda pos, text, owner: shown.setdefault("text", text))

        widget._update_tooltip(12.0, 34.0)

        assert shown["text"].endswith("Value: 123.45")
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
