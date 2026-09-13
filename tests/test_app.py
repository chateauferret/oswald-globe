from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor

from oswald_globe.app import create_window, load_elevation


def test_load_elevation(tmp_path: Path):
    tif_path = tmp_path / "test.tif"
    arr = np.ones((10, 20), dtype=np.float32) * 0.5
    img = Image.fromarray(arr)
    img.save(tif_path)

    elev = load_elevation(tif_path)
    assert elev.shape == (10, 20)
    # 0.5 * 4096.0 - 1024.0 == 1024.0
    np.testing.assert_allclose(elev, 1024.0, atol=1e-3)


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
    assert [action.text() for action in window.menuBar().actions()[0].menu().actions() if action.text()] == [
        "Open",
        "Save",
        "Save As...",
        "Settings",
    ]

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
    assert restored_viewer.gl_widget.show_wireframe is False
    assert restored_viewer.gl_widget.btn_mesh.isChecked() is False
    assert restored_viewer.gl_widget.mesh_color == (1.0, 0.0, 0.0)
    assert restored_viewer.gl_widget.mesh_opacity == 0.25
    assert restored_viewer.gl_widget.graticule_color == (0.0, 1.0, 0.0)
    assert restored_viewer.gl_widget.graticule_opacity == 0.75
    restored_window.close()
