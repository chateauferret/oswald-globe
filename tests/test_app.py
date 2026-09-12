from pathlib import Path
import numpy as np
from PIL import Image
import pytest

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

    window = create_window(
        heightfield=tif_path,
        mesh_min_level=1,
        mesh_max_level=2,
        mesh_threshold=200.0,
    )
    assert window is not None
    assert window.windowTitle() == "globe"
    assert window.centralWidget() is not None
