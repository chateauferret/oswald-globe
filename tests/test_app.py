import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog

from oswald_globe.app import IcosphereProgressDialog, create_window, load_elevation
from oswald_globe.icosphere import IcosphereGrid


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
    assert [action.text() for action in window.menuBar().actions()] == ["File", "Edit", "View"]
    assert [action.text() for action in window.menuBar().actions()[0].menu().actions() if action.text()] == [
        "Open",
        "Save",
        "Save As...",
        "Settings",
    ]
    legend_menu = window.menuBar().actions()[2].menu().actions()[0].menu()
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
    restored_legend_menu = restored_window.menuBar().actions()[2].menu().actions()[0].menu()
    assert restored_viewer.cmap.name == "grayscale"
    assert [action.text() for action in restored_legend_menu.actions() if action.isChecked()] == ["grayscale"]
    assert restored_viewer.gl_widget.show_wireframe is False
    assert restored_viewer.gl_widget.btn_mesh.isChecked() is False
    assert restored_viewer.gl_widget.mesh_color == (1.0, 0.0, 0.0)
    assert restored_viewer.gl_widget.mesh_opacity == 0.25
    assert restored_viewer.gl_widget.graticule_color == (0.0, 1.0, 0.0)
    assert restored_viewer.gl_widget.graticule_opacity == 0.75
    restored_window.close()


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
