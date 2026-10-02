import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QDir, QSettings, Qt, QPointF
from PySide6.QtGui import QColor, QCloseEvent
from PySide6.QtWidgets import QComboBox, QFileDialog, QMessageBox, QProgressBar, QSlider, QSpinBox, QToolTip

import oswald_globe.app as app_module
from oswald_globe.app import IcosphereProgressDialog, create_window, load_elevation
from oswald_globe import globe_main_window
from oswald_globe.globe_main_window import (
    TiffExportProgressDialog,
    TiffImportProgressDialog,
    TiffImportWorker,
    _load_tiff_import_raster,
)
from oswald_globe.icosphere_grid import IcosphereGrid
from oswald_globe.project import Project
from oswald_globe.undo_stack import BrushPaintCommand, SelectionBrushCommand, UndoStack


def _project_window_title(window) -> str:
    return window.windowTitle().replace("[*]", "")


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
        assert window.windowTitle().endswith("[*]")
        assert _project_window_title(window) == "Sea level"
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


def test_tool_loader_discovers_plugins_in_tools_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import importlib

    plugin_path = tmp_path / "demo_tool.py"
    plugin_path.write_text(
        "from oswald_globe.tools.tool import Tool\n"
        "\n"
        "class DemoTool(Tool):\n"
        "    _menu_label = 'Demo'\n"
        "\n"
        "    def __init__(self, parent=None, on_brush_changed=None):\n"
        "        super().__init__(parent)\n"
        "        self._on_brush_changed = on_brush_changed\n"
        "\n"
        "    def create_menu_action(self, tools_menu, action_group, on_selected):\n"
        "        return None\n"
    )

    import oswald_globe.tools as tools_package

    monkeypatch.setattr(tools_package, "__path__", [str(tmp_path)])
    sys.modules.pop("oswald_globe.tools.demo_tool", None)

    discovered = tools_package.discover_tools()
    assert "demo" in discovered

    loaded = tools_package.load_tools(parent=object(), on_brush_changed=lambda: None)
    assert "demo" in loaded
    assert isinstance(loaded["demo"], importlib.import_module("oswald_globe.tools.demo_tool").DemoTool)


def test_filter_loader_discovers_plugins_in_filters_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import importlib

    plugin_path = tmp_path / "demo_filter.py"
    plugin_path.write_text(
        "from oswald_globe.filters.filter import Filter\n"
        "\n"
        "class DemoFilter(Filter):\n"
        "    _menu_label = 'Demo'\n"
        "\n"
        "    def __init__(self, parent=None):\n"
        "        super().__init__(parent)\n"
        "\n"
        "    def create_menu_action(self, filters_menu, action_group, on_selected):\n"
        "        return None\n"
        "\n"
        "    def apply(self, gl_widget):\n"
        "        return None\n"
    )

    import oswald_globe.filters as filters_package

    monkeypatch.setattr(filters_package, "__path__", [str(tmp_path)])
    sys.modules.pop("oswald_globe.filters.demo_filter", None)

    discovered = filters_package.discover_filters()
    assert "demo" in discovered

    loaded = filters_package.load_filters(parent=object())
    assert "demo" in loaded
    assert isinstance(loaded["demo"], importlib.import_module("oswald_globe.filters.demo_filter").DemoFilter)


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


def test_configure_qt_opengl_backend_prefers_glx_on_linux(monkeypatch: pytest.MonkeyPatch):
    class FakeApplication:
        calls = []

        @staticmethod
        def instance():
            return None

        @staticmethod
        def setAttribute(attribute, enabled=True):
            FakeApplication.calls.append((attribute, enabled))

    monkeypatch.setattr(globe_main_window, "QApplication", FakeApplication)
    monkeypatch.setattr(globe_main_window.sys, "platform", "linux")
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.delenv("QT_XCB_GL_INTEGRATION", raising=False)

    globe_main_window._configure_qt_opengl_backend()

    assert os.environ["QT_OPENGL"] == "desktop"
    assert os.environ["QT_XCB_GL_INTEGRATION"] == "xcb_glx"
    assert FakeApplication.calls == [(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)]


def test_configure_qt_runtime_environment_pins_nvidia_egl_vendor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module.sys, "platform", "linux")
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.delenv("QT_XCB_GL_INTEGRATION", raising=False)
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_FILENAMES", raising=False)
    monkeypatch.setattr(
        app_module,
        "_find_nvidia_egl_vendor_library_file",
        lambda: "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )

    app_module._configure_qt_runtime_environment()

    assert os.environ["QT_OPENGL"] == "desktop"
    assert os.environ["QT_XCB_GL_INTEGRATION"] == "xcb_glx"
    assert os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


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
    assert window.windowTitle().endswith("[*]")
    assert _project_window_title(window) == "globe"
    assert window.centralWidget() is not None
    assert [action.text() for action in window.menuBar().actions()] == ["File", "Edit", "Select", "Tools", "Filters", "View"]
    assert window._file_menu is not None
    assert [action.text() for action in window._file_menu.actions() if action.text()] == [
        "New",
        "Open",
        "Save",
        "Save As...",
        "Import",
        "Export",
        "Export As...",
        "Settings",
    ]
    assert window._edit_menu is not None
    assert [action.text() for action in window._edit_menu.actions()] == ["Undo", "Redo"]
    assert [action.isEnabled() for action in window._edit_menu.actions()] == [False, False]
    assert window._select_menu is not None
    assert [action.text() for action in window._select_menu.actions()] == ["All", "None", "Invert"]
    assert window._tools_menu is not None
    assert [action.text() for action in window._tools_menu.actions()] == ["Navigate", "Select", "Paint"]
    assert [action.text() for action in window._tools_menu.actions() if action.isChecked()] == ["Navigate"]
    assert window._filters_menu is not None
    assert [action.text() for action in window._filters_menu.actions()] == ["Fill"]

    assert window._legend_menu is not None
    assert window._legend_menu.toolTipsVisible() is True
    legend_menu = window._legend_menu
    window._populate_legend_menu()
    assert [action.text() for action in legend_menu.actions()] == ["grayscale", "topography"]
    assert all(not action.icon().isNull() for action in legend_menu.actions())
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

    window._mark_project_clean()
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


def test_new_project_resets_to_sea_level_when_clean(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.ones((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    try:
        original_viewer = window.viewer
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("warning should not be shown")),
        )

        window.new_project()

        assert window.viewer is not original_viewer
        assert _project_window_title(window) == "Sea level"
        assert window._heightfield is None
        assert window._save_path is None
        np.testing.assert_allclose(window.viewer.data, 0.0)
        assert window._undo_stack.is_clean() is True
        assert window.isWindowModified() is False
    finally:
        window.close()


def test_new_project_cancel_keeps_existing_state(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        window._undo_stack.push(DummyCommand())
        original_viewer = window.viewer
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
        )

        window.new_project()

        assert window.viewer is original_viewer
        assert window._undo_stack.is_clean() is False
        assert window.isWindowModified() is True
    finally:
        window.close()


def test_new_project_save_then_resets(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        save_path = tmp_path / "saved_project.ogp"
        saved = []
        window._undo_stack.push(DummyCommand())
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Save,
        )
        monkeypatch.setattr(
            window,
            "save_heightfield",
            lambda: saved.append(True) or setattr(window, "_save_path", save_path) or True,
        )

        window.new_project()

        assert saved == [True]
        assert _project_window_title(window) == "Sea level"
        assert window._heightfield is None
        assert window._save_path is None
        assert window._undo_stack.is_clean() is True
        assert window.isWindowModified() is False
    finally:
        window.close()


def test_new_project_discard_then_resets_without_saving(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        saved = []
        original_viewer = window.viewer
        window._undo_stack.push(DummyCommand())
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
        )
        monkeypatch.setattr(window, "save_heightfield", lambda: saved.append(True) or True)

        window.new_project()

        assert saved == []
        assert window.viewer is not original_viewer
        assert _project_window_title(window) == "Sea level"
        assert window._undo_stack.is_clean() is True
        assert window.isWindowModified() is False
    finally:
        window.close()


def test_close_event_cancel_keeps_window_open(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        window._undo_stack.push(DummyCommand())
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
        )

        event = QCloseEvent()
        window.closeEvent(event)

        assert event.isAccepted() is False
        assert window.isWindowModified() is True
    finally:
        window._mark_project_clean()
        window.close()


def test_close_event_save_then_accepts(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        window._undo_stack.push(DummyCommand())
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Save,
        )
        monkeypatch.setattr(window, "save_heightfield", lambda: True)
        sync_calls = []
        monkeypatch.setattr(window, "_save_grid_settings", lambda: sync_calls.append("grid"))
        monkeypatch.setattr(window._settings, "sync", lambda: sync_calls.append("sync"))

        event = QCloseEvent()
        window.closeEvent(event)

        assert event.isAccepted() is True
        assert sync_calls == ["grid", "sync"]
    finally:
        window._mark_project_clean()
        window.close()


def test_close_event_discard_then_accepts_without_saving(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        class DummyCommand:
            def redo(self) -> None:
                pass

            def undo(self) -> None:
                pass

        saved = []
        window._undo_stack.push(DummyCommand())
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
        )
        monkeypatch.setattr(window, "save_heightfield", lambda: saved.append(True) or True)
        sync_calls = []
        monkeypatch.setattr(window, "_save_grid_settings", lambda: sync_calls.append("grid"))
        monkeypatch.setattr(window._settings, "sync", lambda: sync_calls.append("sync"))

        event = QCloseEvent()
        window.closeEvent(event)

        assert saved == []
        assert event.isAccepted() is True
        assert sync_calls == ["grid", "sync"]
    finally:
        window._mark_project_clean()
        window.close()


def test_paint_tool_selection_shows_options_and_navigate_disposes(qapp):
    window = create_window()
    try:
        assert window._tools_menu is not None
        tools_menu = window._tools_menu
        navigate_action, select_action, paint_action = tools_menu.actions()
        paint_tool = window._tools["paint"]
        select_tool = window._tools["select"]

        assert navigate_action.isChecked() is True
        assert select_action.isChecked() is False
        assert paint_action.isChecked() is False
        assert paint_tool.options_dialog is None
        assert select_tool.options_dialog is None

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

        select_action.trigger()
        assert select_action.isChecked() is True
        assert navigate_action.isChecked() is False
        assert paint_action.isChecked() is False
        assert paint_tool.options_dialog is None
        assert select_tool.options_dialog is not None
        assert select_tool.options_dialog.windowTitle() == "Select Tool Options"

        sliders = select_tool.options_dialog.findChildren(QSlider)
        ranges = sorted((slider.minimum(), slider.maximum()) for slider in sliders if slider.isVisible())
        assert ranges == [(0, 100), (0, 1000)]
        spin_boxes = select_tool.options_dialog.findChildren(QSpinBox)
        spin_ranges = sorted((spin.minimum(), spin.maximum()) for spin in spin_boxes if spin.isVisible())
        assert spin_ranges == [(0, 100), (0, 1000)]
        mode_combo = select_tool.options_dialog.findChild(QComboBox)
        assert mode_combo is not None
        assert mode_combo.isHidden() is True
        assert select_tool.options_dialog.radius_spin.value() == 100
        assert select_tool.options_dialog.falloff_spin.value() == 50

        select_tool.options_dialog.radius_slider.setValue(321)
        select_tool.options_dialog.falloff_slider.setValue(12)
        assert select_tool.radius_km() == 321
        assert select_tool.falloff_percent() == 12

        navigate_action.trigger()
        assert select_tool.options_dialog is None
        assert select_tool.radius_km() == 321
        assert select_tool.falloff_percent() == 12
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


def test_brush_command_applies_without_selection_when_has_selection_is_false():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    original = np.linspace(0.0, 100.0, grid.vertex_count(), dtype=np.float64)
    grid.set_layer("elevation", original.copy())
    grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values[:] = 0.0

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0

    command = BrushPaintCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=8000.0,
        falloff_percent=50.0,
        paint_value=8,
        paint_mode="Add",
        mesh_grid=grid,
        apply_vertex_values=lambda indices, values: grid.get_layer("elevation").values.__setitem__(indices, values),
    )

    command.redo()

    assert grid.has_selection is False
    np.testing.assert_allclose(
        grid.get_layer("elevation").values[command.vertex_indices],
        command.after_values,
    )


def test_brush_command_multiplies_effect_by_selection_status():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    original = np.linspace(0.0, 100.0, grid.vertex_count(), dtype=np.float64)
    grid.set_layer("elevation", original.copy())
    reference_grid = IcosphereGrid()
    reference_grid.subdivide_uniform(1)
    reference_grid.set_layer("elevation", original.copy())

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0
    radius_km = 8000.0
    falloff_percent = 50.0

    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 0.0

    center = grid.vertices[target_index]
    distances = np.arccos(np.clip(grid.vertices @ center, -1.0, 1.0))
    outer_radius = np.deg2rad(min(89.0, radius_km / 111.0))
    selected_indices = np.flatnonzero(distances <= outer_radius + 1e-9)
    assert selected_indices.size > 0

    selection.values[selected_indices] = np.linspace(0.25, 1.0, selected_indices.size, dtype=np.float64)
    assert grid.has_selection is True

    reference_command = BrushPaintCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=radius_km,
        falloff_percent=falloff_percent,
        paint_value=8,
        paint_mode="Add",
        mesh_grid=reference_grid,
        apply_vertex_values=lambda indices, values: reference_grid.get_layer("elevation").values.__setitem__(indices, values),
    )
    command = BrushPaintCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=radius_km,
        falloff_percent=falloff_percent,
        paint_value=8,
        paint_mode="Add",
        mesh_grid=grid,
        apply_vertex_values=lambda indices, values: grid.get_layer("elevation").values.__setitem__(indices, values),
    )

    command.redo()

    selection_weights = selection.values[command.vertex_indices]
    unselected_mask = selection_weights == 0.0
    selected_mask = selection_weights > 0.0

    assert np.any(selected_mask)
    expected_after_values = command.before_values + selection_weights * (reference_command.after_values - command.before_values)
    np.testing.assert_allclose(command.after_values, expected_after_values, atol=1e-6)
    if np.any(unselected_mask):
        np.testing.assert_allclose(command.after_values[unselected_mask], command.before_values[unselected_mask])


def test_selection_brush_command_marks_vertices_with_soft_falloff():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 0.0

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0
    radius_km = 8000.0
    falloff_percent = 50.0

    command = SelectionBrushCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=radius_km,
        falloff_percent=falloff_percent,
        replace_existing=False,
        erase_selection=False,
        mesh_grid=grid,
        apply_selection_values=lambda indices, values: selection.values.__setitem__(indices, values),
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

    np.testing.assert_allclose(selection.values[inner_mask], 1.0)
    expected_transition = (outer_radius - distances[transition_mask]) / (outer_radius - inner_radius)
    np.testing.assert_allclose(selection.values[transition_mask], expected_transition)

    command.undo()
    np.testing.assert_allclose(selection.values, 0.0)


def test_selection_brush_command_can_replace_existing_selection():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 1.0

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0

    command = SelectionBrushCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=8000.0,
        falloff_percent=50.0,
        replace_existing=True,
        erase_selection=False,
        mesh_grid=grid,
        apply_selection_values=lambda indices, values: selection.values.__setitem__(indices, values),
    )

    command.redo()

    assert command.vertex_indices.shape == (grid.vertex_count(),)
    assert np.any(selection.values == 0.0)
    assert np.max(selection.values) == pytest.approx(1.0)

    command.undo()
    np.testing.assert_allclose(selection.values, 1.0)


def test_selection_brush_command_can_erase_without_clearing_unaffected_vertices():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 1.0

    lat_deg, lon_deg = grid.vertex_lat_lon()
    target_index = 0

    command = SelectionBrushCommand(
        target_lat_deg=float(lat_deg[target_index]),
        target_lon_deg=float(lon_deg[target_index]),
        radius_km=8000.0,
        falloff_percent=50.0,
        replace_existing=False,
        erase_selection=True,
        mesh_grid=grid,
        apply_selection_values=lambda indices, values: selection.values.__setitem__(indices, values),
    )

    command.redo()

    assert np.any(command.after_values == 0.0)
    unaffected = np.setdiff1d(np.arange(grid.vertex_count(), dtype=np.intp), command.vertex_indices)
    if unaffected.size:
        np.testing.assert_allclose(selection.values[unaffected], 1.0)

    command.undo()
    np.testing.assert_allclose(selection.values, 1.0)


def test_selection_layer_command_bulk_updates_and_undo():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = np.linspace(0.0, 1.0, grid.vertex_count(), dtype=np.float64)

    from oswald_globe.undo_stack import SelectionLayerCommand

    target = np.ones(grid.vertex_count(), dtype=np.float64)
    command = SelectionLayerCommand(
        mesh_grid=grid,
        after_values=target,
        apply_selection_values=lambda indices, values: selection.values.__setitem__(indices, values),
    )

    command.redo()
    np.testing.assert_allclose(selection.values, 1.0)

    command.undo()
    np.testing.assert_allclose(selection.values, np.linspace(0.0, 1.0, grid.vertex_count(), dtype=np.float64))


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


def test_select_tool_applies_selection_and_shows_brush_hover(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")

        updates = []
        monkeypatch.setattr(widget, "update", lambda: updates.append(True))
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

        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_select_command)

        class SelectEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.LeftButton

            def buttons(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

        widget.mouseMoveEvent(SelectEvent(12.0, 34.0))
        widget.mouseReleaseEvent(SelectEvent(12.0, 34.0))

        selection = widget.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values
        assert widget._hover_target_lat == pytest.approx(0.5)
        assert widget._hover_target_lon == pytest.approx(-1.0)
        assert widget.cursor().shape() == Qt.CursorShape.BlankCursor
        assert np.max(selection) > 0.0
        assert updates
    finally:
        window.close()


def test_select_tool_alt_hover_draws_only_outer_brush_ring(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")
        widget._mouse_over_globe = True
        widget.line_program = 1
        widget.paint_angular_radius = 1.0
        widget.paint_dropoff_percent = 50.0
        widget._select_alt_mode = True

        ring_calls = []

        monkeypatch.setattr(widget, "_brush_ring_vertices", lambda radius: ring_calls.append(radius) or np.zeros((2, 3), dtype=np.float32))
        monkeypatch.setattr("oswald_globe.globe_widget.glGenBuffers", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr("oswald_globe.globe_widget.glBindBuffer", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glBufferData", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glUseProgram", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glGetUniformLocation", lambda *args, **kwargs: 0)
        monkeypatch.setattr("oswald_globe.globe_widget.glUniformMatrix4fv", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glUniform3f", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glUniform1f", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glGetAttribLocation", lambda *args, **kwargs: -1)
        monkeypatch.setattr("oswald_globe.globe_widget.glLineWidth", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glDepthMask", lambda *args, **kwargs: None)
        monkeypatch.setattr("oswald_globe.globe_widget.glDrawArrays", lambda *args, **kwargs: None)

        widget._draw_brush_overlay(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))

        assert ring_calls == [pytest.approx(widget.paint_angular_radius)]
    finally:
        window.close()


def test_select_tool_mouse_down_replaces_existing_selection_without_shift(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")
        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_select_command)

        selection = widget.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        selection.values[:] = 1.0

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

        class SelectPressEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

            def modifiers(self):
                return Qt.KeyboardModifier.NoModifier

        widget.mousePressEvent(SelectPressEvent(12.0, 34.0))

        assert np.any(selection.values == 0.0)
        assert np.max(selection.values) == pytest.approx(1.0)
    finally:
        window.close()


def test_select_tool_mouse_down_with_shift_adds_to_existing_selection(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")
        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_select_command)

        selection = widget.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        selection.values[:] = 0.0
        lat_deg, lon_deg = widget.mesh_grid.vertex_lat_lon()
        target_index = 0
        center = widget.mesh_grid.vertices[target_index]
        distances = np.arccos(np.clip(widget.mesh_grid.vertices @ center, -1.0, 1.0))
        far_index = int(np.argmax(distances))
        selection.values[far_index] = 0.5

        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {
                "target_lat": np.deg2rad(float(lat_deg[target_index])),
                "target_lon": np.deg2rad(float(lon_deg[target_index])),
                "value": 0.0,
                "lat_deg": float(lat_deg[target_index]),
                "lon_deg": float(lon_deg[target_index]),
            },
        )

        class SelectShiftPressEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

            def modifiers(self):
                return Qt.KeyboardModifier.ShiftModifier

        widget.mousePressEvent(SelectShiftPressEvent(12.0, 34.0))

        assert selection.values[far_index] == pytest.approx(0.5)
        assert np.max(selection.values) == pytest.approx(1.0)
    finally:
        window.close()


def test_select_tool_mouse_down_with_alt_erases_without_clearing_other_selection(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")
        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_select_command)

        selection = widget.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        selection.values[:] = 1.0
        lat_deg, lon_deg = widget.mesh_grid.vertex_lat_lon()
        target_index = 0
        center = widget.mesh_grid.vertices[target_index]
        distances = np.arccos(np.clip(widget.mesh_grid.vertices @ center, -1.0, 1.0))
        far_index = int(np.argmax(distances))

        monkeypatch.setattr(
            "oswald_globe.globe_widget.unproject_point",
            lambda *args, **kwargs: {
                "target_lat": np.deg2rad(float(lat_deg[target_index])),
                "target_lon": np.deg2rad(float(lon_deg[target_index])),
                "value": 0.0,
                "lat_deg": float(lat_deg[target_index]),
                "lon_deg": float(lon_deg[target_index]),
            },
        )

        class SelectAltPressEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def button(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

            def modifiers(self):
                return Qt.KeyboardModifier.AltModifier

        widget.mousePressEvent(SelectAltPressEvent(12.0, 34.0))

        assert np.any(selection.values == 0.0)
        assert selection.values[far_index] == pytest.approx(1.0)
    finally:
        window.close()


def test_select_tool_alt_drag_erases_selection_under_outer_circle(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("select")
        widget.set_undo_stack(UndoStack())
        widget.set_brush_command_factory(window._create_select_command)

        selection = widget.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        selection.values[:] = 1.0

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

        class SelectAltDragEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def buttons(self):
                return Qt.MouseButton.LeftButton

            def position(self):
                return self._pos

            def modifiers(self):
                return Qt.KeyboardModifier.AltModifier

        widget.mouseMoveEvent(SelectAltDragEvent(12.0, 34.0))

        assert np.any(selection.values == 0.0)
    finally:
        window.close()


def test_select_menu_actions_update_selection_and_support_undo(qapp):
    window = create_window()
    try:
        selection = window.viewer.mesh_grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
        selection.values[:] = np.linspace(0.0, 1.0, selection.values.size, dtype=np.float64)
        original = selection.values.copy()

        window.select_all_vertices()
        np.testing.assert_allclose(selection.values, 1.0)

        window._undo_stack.undo()
        np.testing.assert_allclose(selection.values, original)

        window.select_no_vertices()
        np.testing.assert_allclose(selection.values, 0.0)

        window._undo_stack.undo()
        np.testing.assert_allclose(selection.values, original)

        window.invert_vertex_selection()
        np.testing.assert_allclose(selection.values, 1.0 - original)

        window._undo_stack.undo()
        np.testing.assert_allclose(selection.values, original)
    finally:
        window.close()


def test_auto_spin_blocks_globe_mouse_interactions(qapp, monkeypatch: pytest.MonkeyPatch):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.anim_timer.stop()
        widget.is_auto_spin = True

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

        class HoverEvent:
            def __init__(self, x: float, y: float):
                self._pos = QPointF(x, y)

            def buttons(self):
                return Qt.MouseButton.NoButton

            def position(self):
                return self._pos

        class PressEvent(HoverEvent):
            def button(self):
                return Qt.MouseButton.LeftButton

        class WheelDelta:
            def y(self):
                return 120

        class WheelEvent(HoverEvent):
            def angleDelta(self):
                return WheelDelta()

        start_lon = widget.center_lon
        start_lat = widget.center_lat
        start_zoom = widget.zoom

        widget.mouseMoveEvent(HoverEvent(12.0, 34.0))
        assert widget.cursor().shape() == Qt.CursorShape.ForbiddenCursor

        widget.mousePressEvent(PressEvent(12.0, 34.0))
        assert widget.is_dragging is False

        widget.mouseMoveEvent(PressEvent(22.0, 44.0))
        widget.mouseReleaseEvent(PressEvent(22.0, 44.0))
        widget.mouseDoubleClickEvent(PressEvent(22.0, 44.0))
        widget.wheelEvent(WheelEvent(22.0, 44.0))

        assert widget.cursor().shape() == Qt.CursorShape.ForbiddenCursor
        assert widget.center_lon == pytest.approx(start_lon)
        assert widget.center_lat == pytest.approx(start_lat)
        assert widget.zoom == pytest.approx(start_zoom)
    finally:
        window.close()


def test_auto_spin_hides_brush_overlay(qapp):
    window = create_window()
    try:
        widget = window.viewer.gl_widget
        widget.set_tool_mode("paint")
        widget.is_auto_spin = True
        widget._mouse_over_globe = True
        widget.paint_angular_radius = 1.0
        widget.line_program = 1
        widget.brush_vertex_count = 99

        widget._draw_brush_overlay(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))

        assert widget.brush_vertex_count == 99
    finally:
        window.close()


def test_open_heightfield_dialog_configuration(qapp):
    window = create_window()

    dialog = window._create_open_heightfield_dialog()

    assert dialog.acceptMode() == dialog.AcceptMode.AcceptOpen
    assert dialog.fileMode() == dialog.FileMode.ExistingFile
    assert dialog.nameFilters()[0].startswith("Oswald Globe Projects (")
    assert "*.ogp" in dialog.nameFilters()[0]
    assert dialog.nameFilters()[1].startswith("NumPy Arrays (")
    assert "*.npy" in dialog.nameFilters()[1]
    assert "*.npz" in dialog.nameFilters()[1]

    window.close()


def test_import_tiff_dialog_configuration(qapp):
    window = create_window()

    dialog = window._create_import_tiff_dialog()

    assert dialog.acceptMode() == dialog.AcceptMode.AcceptOpen
    assert dialog.fileMode() == dialog.FileMode.ExistingFile
    assert dialog.nameFilters()[0].startswith("TIFF Images (")
    assert "*.tif" in dialog.nameFilters()[0]
    assert "*.tiff" in dialog.nameFilters()[0]

    window.close()


def test_open_heightfield_dialog_excludes_tiff(qapp, tmp_path: Path):
    tif_path = tmp_path / "globe.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(tif_path)
    window = create_window(heightfield=tif_path, mesh_min_level=1, mesh_max_level=2, mesh_threshold=200.0)

    dialog = window._create_open_heightfield_dialog()

    assert "*.tif" not in dialog.nameFilters()[0]

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


def test_available_legend_sources_merges_resources_and_filesystem(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from oswald_globe.globe_main_window import GlobeMainWindow

    legends_dir = tmp_path / "legends"
    legends_dir.mkdir()
    (legends_dir / "topography.txt").write_text(
        "{'red': ((0.0, 0.0, 0.0),), 'green': ((0.0, 0.0, 0.0),), 'blue': ((0.0, 0.0, 0.0),)}",
        encoding="utf-8",
    )
    monkeypatch.setattr("oswald_globe.globe_main_window.LEGENDS_DIR", legends_dir)
    monkeypatch.setattr(
        QDir,
        "entryList",
        lambda self, _patterns, _filters, _sort: ["grayscale.txt"],
    )

    window = create_window()
    try:
        sources = window._available_legend_sources()
        assert ":/legends/grayscale.txt" in sources
        assert str(legends_dir / "topography.txt") in sources
        assert sorted(window._legend_name(source) for source in sources) == ["grayscale", "topography"]
    finally:
        window.close()


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
    assert _project_window_title(window) == "original"

    window.close()


def test_open_heightfield_loads_selected_numpy_array(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_path = tmp_path / "original.tif"
    replacement_path = tmp_path / "replacement.npy"

    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(original_path)
    replacement_data = np.linspace(-1000, 3000, 64 * 128, dtype=np.float32).reshape(64, 128)
    np.save(replacement_path, replacement_data)

    window = create_window(heightfield=original_path, mesh_min_level=1, mesh_max_level=3, mesh_threshold=50.0)
    original_viewer = window.viewer
    original_mesh = original_viewer.mesh_grid
    built_mesh = IcosphereGrid.from_equirectangular(
        replacement_data,
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
    assert _project_window_title(window) == "replacement"
    assert window.viewer is not original_viewer
    assert window.viewer.mesh_grid is not None
    assert window.viewer.mesh_grid is not original_mesh
    assert window.viewer.mesh_grid is built_mesh
    assert window.viewer.data.shape == replacement_data.shape

    window.close()


def test_project_round_trip_preserves_mesh_geometry_layers_and_metadata(tmp_path: Path):
    raster = np.arange(32, dtype=np.float32).reshape(4, 8)
    grid = IcosphereGrid.from_equirectangular(raster, min_level=1, max_level=2, threshold=5.0)
    grid.add_layer(
        "temperature",
        np.linspace(-10.0, 25.0, grid.vertex_count(), dtype=np.float64),
        description="Surface temperature",
        visible=False,
        opacity=42.0,
        legend="grayscale",
        z_index=3,
    )
    grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values[:] = np.linspace(
        0.0,
        1.0,
        grid.vertex_count(),
        dtype=np.float64,
    )
    metadata = {"legend_source": ":/legends/grayscale.txt", "author": "tests"}
    project_path = tmp_path / "round_trip.ogp"

    Project(raster=raster, grid=grid, metadata=metadata).save(project_path)
    loaded = Project.load(project_path)

    np.testing.assert_allclose(loaded.raster, raster)
    assert loaded.metadata == metadata
    np.testing.assert_allclose(loaded.grid.vertices, grid.vertices)
    np.testing.assert_array_equal(loaded.grid.layer_names(), grid.layer_names())
    np.testing.assert_allclose(
        loaded.grid.get_layer("elevation").values,
        grid.get_layer("elevation").values,
    )
    np.testing.assert_allclose(
        loaded.grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values,
        grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME).values,
    )
    loaded_temperature = loaded.grid.get_layer("temperature")
    original_temperature = grid.get_layer("temperature")
    np.testing.assert_allclose(loaded_temperature.values, original_temperature.values)
    assert loaded_temperature.description == "Surface temperature"
    assert loaded_temperature.visible is False
    assert loaded_temperature.opacity == pytest.approx(42.0)
    assert loaded_temperature.legend is not None
    assert loaded_temperature.legend.name == "grayscale"
    assert loaded_temperature.z_index == 3


def test_create_window_loads_project_file_without_rebuilding_mesh(qapp, tmp_path: Path):
    raster = np.arange(32, dtype=np.float32).reshape(4, 8)
    grid = IcosphereGrid.from_equirectangular(raster, min_level=1, max_level=2, threshold=5.0)
    grid.get_layer("elevation").values[:] = np.linspace(
        -50.0, 75.0, grid.vertex_count(), dtype=np.float64
    )
    project_path = tmp_path / "saved_project.ogp"
    Project(
        raster=raster,
        grid=grid,
        metadata={"legend_source": ":/legends/grayscale.txt"},
    ).save(project_path)

    window = create_window(heightfield=project_path, mesh_min_level=1, mesh_max_level=5, mesh_threshold=999.0)
    try:
        assert window._heightfield == project_path.resolve()
        assert window._save_path == project_path.resolve()
        assert _project_window_title(window) == "saved_project"
        assert window.viewer.cmap.name == "grayscale"
        assert window.viewer.mesh_grid is not None
        np.testing.assert_allclose(window.viewer.data, raster)
        np.testing.assert_allclose(window.viewer.mesh_grid.vertices, grid.vertices)
        np.testing.assert_allclose(
            window.viewer.mesh_grid.get_layer("elevation").values,
            grid.get_layer("elevation").values,
        )
    finally:
        window.close()


def test_open_heightfield_loads_selected_project_file(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    raster = np.arange(32, dtype=np.float32).reshape(4, 8)
    project_grid = IcosphereGrid.from_equirectangular(raster, min_level=1, max_level=2, threshold=5.0)
    project_path = tmp_path / "saved_project.ogp"
    Project(raster=raster, grid=project_grid, metadata={}).save(project_path)

    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=4, mesh_threshold=999.0)
    try:
        original_viewer = window.viewer

        class AcceptDialog:
            def exec(self):
                return QFileDialog.DialogCode.Accepted

            def selectedFiles(self):
                return [str(project_path)]

        started = []
        monkeypatch.setattr(window, "_create_open_heightfield_dialog", lambda: AcceptDialog())
        monkeypatch.setattr(window, "_start_heightfield_load", lambda *args: started.append(args))

        window.open_heightfield()

        assert started == []
        assert window.viewer is not original_viewer
        assert window._heightfield == project_path.resolve()
        assert window._save_path == project_path.resolve()
        np.testing.assert_allclose(window.viewer.data, raster)
        assert window.viewer.mesh_grid is not None
        np.testing.assert_allclose(window.viewer.mesh_grid.vertices, project_grid.vertices)
        np.testing.assert_allclose(
            window.viewer.mesh_grid.get_layer("elevation").values,
            project_grid.get_layer("elevation").values,
        )
    finally:
        window.close()


def test_save_heightfield_writes_project_file_with_current_mesh_state(qapp, tmp_path: Path):
    window = create_window(mesh_min_level=1, mesh_max_level=2)
    try:
        widget = window.viewer.gl_widget
        assert widget.mesh_grid is not None
        widget.mesh_grid.get_layer("elevation").values[:] = np.linspace(
            10.0, 20.0, widget.mesh_grid.vertex_count(), dtype=np.float64
        )
        widget.mesh_grid.add_layer(
            "humidity",
            np.linspace(0.0, 1.0, widget.mesh_grid.vertex_count(), dtype=np.float64),
            description="Relative humidity",
        )
        window.set_legend(":/legends/grayscale.txt")

        project_path = tmp_path / "saved_project.ogp"
        window._save_heightfield(project_path)

        loaded = Project.load(project_path)
        np.testing.assert_allclose(loaded.raster, window.viewer.data)
        assert loaded.metadata["legend_source"] == ":/legends/grayscale.txt"
        np.testing.assert_allclose(
            loaded.grid.get_layer("elevation").values,
            widget.mesh_grid.get_layer("elevation").values,
        )
        np.testing.assert_allclose(
            loaded.grid.get_layer("humidity").values,
            widget.mesh_grid.get_layer("humidity").values,
        )
    finally:
        window._mark_project_clean()
        window.close()


def test_import_tiff_dispatches_to_background_worker(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tiff_path = tmp_path / "world.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.uint8)).save(tiff_path)
    window = create_window(mesh_min_level=1, mesh_max_level=2)

    class AcceptDialog:
        def exec(self):
            return QFileDialog.DialogCode.Accepted

        def selectedFiles(self):
            return [str(tiff_path)]

    monkeypatch.setattr(window, "_create_import_tiff_dialog", lambda: AcceptDialog())
    started = []
    monkeypatch.setattr(window, "_start_tiff_import", lambda path: started.append(Path(path)))

    window.import_tiff()

    assert started == [tiff_path]
    window.close()


def test_load_tiff_import_raster_rescales_to_2_to_1_and_maps_luminance(tmp_path: Path):
    tiff_path = tmp_path / "square.tif"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :2] = 0
    rgb[:, 2:] = 255
    Image.fromarray(rgb, mode="RGB").save(tiff_path)

    raster = _load_tiff_import_raster(tiff_path)

    assert raster.shape == (4, 8)
    assert float(raster.min()) == -32767.0
    assert float(raster.max()) == 32767.0


def test_load_tiff_import_raster_preserves_16bit_grayscale_levels(tmp_path: Path):
    tiff_path = tmp_path / "grayscale16.tif"
    grayscale = np.array(
        [
            [0, 16384, 32768, 65535],
            [65535, 32768, 16384, 0],
        ],
        dtype=np.uint16,
    )
    Image.fromarray(grayscale).save(tiff_path)

    raster = _load_tiff_import_raster(tiff_path)

    expected = grayscale.astype(np.float32) / 65535.0 * 65534.0 - 32767.0
    np.testing.assert_allclose(raster, expected, atol=1.0)
    assert np.unique(raster).size > 2


def test_import_tiff_completion_replaces_viewer_with_imported_adaptive_grid(qapp, tmp_path: Path):
    tiff_path = tmp_path / "world.tif"
    grayscale = np.tile(np.array([[0, 255]], dtype=np.uint8), (8, 1))
    Image.fromarray(grayscale, mode="L").save(tiff_path)
    window = create_window(mesh_min_level=1, mesh_max_level=1)
    try:
        raster = _load_tiff_import_raster(tiff_path)
        original_grid = window.viewer.gl_widget.mesh_grid
        imported_grid = IcosphereGrid.from_equirectangular(
            raster,
            min_level=window._mesh_min_level,
            max_level=window._mesh_max_level,
            threshold=window._mesh_threshold,
        )
        window._pending_import_path = tiff_path

        window._on_tiff_import_completed((raster, imported_grid))

        displayed_grid = window.viewer.gl_widget.mesh_grid
        assert displayed_grid is not None
        assert displayed_grid is imported_grid
        assert displayed_grid is not original_grid
        np.testing.assert_allclose(window.viewer.data, raster)
        np.testing.assert_allclose(
            displayed_grid.get_layer("elevation").values,
            imported_grid.get_layer("elevation").values,
        )
    finally:
        window.close()


def test_tiff_import_worker_builds_adaptive_icosphere_from_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tiff_path = tmp_path / "world.tif"
    grayscale = np.arange(16, dtype=np.uint8).reshape(4, 4)
    Image.fromarray(grayscale, mode="L").save(tiff_path)

    observed = {}
    built_grid = IcosphereGrid()

    def fake_from_equirectangular(arr, *, min_level, max_level, threshold, progress_callback, is_cancelled):
        observed["shape"] = arr.shape
        observed["min_level"] = min_level
        observed["max_level"] = max_level
        observed["threshold"] = threshold
        progress_callback("creating-faces", 1, 4)
        progress_callback("balancing-faces", 1, 2)
        progress_callback("populating-faces", 3, 6)
        assert is_cancelled() is False
        return built_grid

    monkeypatch.setattr(IcosphereGrid, "from_equirectangular", staticmethod(fake_from_equirectangular))

    worker = TiffImportWorker(tiff_path, min_level=2, max_level=5, threshold=123.0)
    progress = []
    messages = []
    completed = []
    worker.progressChanged.connect(lambda done, total: progress.append((done, total)))
    worker.progressMessageChanged.connect(messages.append)
    worker.completed.connect(completed.append)

    worker.run()

    assert observed == {
        "shape": (4, 8),
        "min_level": 2,
        "max_level": 5,
        "threshold": 123.0,
    }
    assert messages[:2] == ["Loading TIFF file", "Mapping TIFF onto icosphere data"]
    assert "Creating icosphere faces" in messages
    assert "Balancing icosphere faces" in messages
    assert "Populating globe data" in messages
    assert completed
    imported_raster, imported_grid = completed[0]
    assert imported_grid is built_grid
    assert imported_raster.shape == (4, 8)
    assert progress[-1] == (100, 100)


def test_export_tiff_rasterizes_icosphere_to_equirectangular_map(qapp, tmp_path: Path):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    grid = window.viewer.gl_widget.mesh_grid
    assert grid is not None
    grid.get_layer("elevation").values[:] = 123.5
    output_path = tmp_path / "export.tif"

    window._export_tiff(output_path)

    with Image.open(output_path) as exported:
        assert exported.mode.startswith("I;16")
        raster = np.asarray(exported)
    assert raster.shape == (2048, 4096)
    assert raster.dtype == np.uint16
    expected = int(round((123.5 + 32767.0) * (65535.0 / 65534.0)))
    assert int(raster.min()) == expected
    assert int(raster.max()) == expected
    window.close()


def test_export_tiff_maps_signed_16bit_range_to_grayscale(qapp, tmp_path: Path):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    grid = window.viewer.gl_widget.mesh_grid
    assert grid is not None

    low_path = tmp_path / "export_low.tif"
    high_path = tmp_path / "export_high.tif"

    grid.get_layer("elevation").values[:] = -32767.0
    window._export_tiff(low_path)
    with Image.open(low_path) as exported_low:
        raster_low = np.asarray(exported_low)
    assert raster_low.dtype == np.uint16
    assert int(raster_low.min()) == 0
    assert int(raster_low.max()) == 0

    grid.get_layer("elevation").values[:] = 32767.0
    window._export_tiff(high_path)
    with Image.open(high_path) as exported_high:
        raster_high = np.asarray(exported_high)
    assert raster_high.dtype == np.uint16
    assert int(raster_high.min()) == 65535
    assert int(raster_high.max()) == 65535
    window.close()


def test_export_tiff_is_readable_by_rasterio_with_expected_range(qapp, tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")

    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    grid = window.viewer.gl_widget.mesh_grid
    assert grid is not None
    grid.get_layer("elevation").values[:] = 123.5
    output_path = tmp_path / "export.tif"

    window._export_tiff(output_path)

    with rasterio.open(output_path) as dataset:
        raster = dataset.read(1)
        assert dataset.width == 4096
        assert dataset.height == 2048
        assert dataset.count == 1
        assert dataset.dtypes[0] == "uint16"
        assert dataset.crs is not None
    expected = int(round((123.5 + 32767.0) * (65535.0 / 65534.0)))
    assert int(raster.min()) == expected
    assert int(raster.max()) == expected
    window.close()


def test_export_tiff_starts_background_export(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    try:
        started = []
        export_path = tmp_path / "export.tif"
        window._export_path = export_path
        monkeypatch.setattr(window, "_start_tiff_export", lambda path: started.append(Path(path)))

        window.export_tiff()

        assert started == [export_path]
    finally:
        window.close()


def test_export_tiff_as_starts_background_export(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.tif"
    Image.fromarray(np.zeros((32, 64), dtype=np.float32)).save(source_path)
    window = create_window(heightfield=source_path, mesh_min_level=1, mesh_max_level=2)
    try:
        started = []
        monkeypatch.setattr(window, "_start_tiff_export", lambda path: started.append(Path(path)))
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(tmp_path / "async_export"), "TIFF Image (*.tif *.tiff)"),
        )

        window.export_tiff_as()

        assert started == [tmp_path / "async_export.tif"]
        assert window._export_path == tmp_path / "async_export.tif"
    finally:
        window.close()


def test_tiff_export_progress_dialog_updates_percentage(qapp):
    dialog = TiffExportProgressDialog()
    try:
        dialog.update_progress(42, 100)
        progress_bar = dialog.findChild(QProgressBar)
        assert progress_bar is not None
        assert progress_bar.value() == 42
        assert progress_bar.maximum() == 100
    finally:
        dialog.close()


def test_tiff_import_progress_dialog_updates_percentage(qapp):
    dialog = TiffImportProgressDialog()
    try:
        dialog.update_progress(42, 100)
        progress_bar = dialog.findChild(QProgressBar)
        assert progress_bar is not None
        assert progress_bar.value() == 42
        assert progress_bar.maximum() == 100
    finally:
        dialog.close()
