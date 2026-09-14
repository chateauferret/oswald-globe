import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None

from oswald_globe.colormap import load_topo_cmap
from oswald_globe.globe_viewer import GlobeViewer, globe_viewer
from oswald_globe.icosphere import IcosphereGrid


def test_globe_viewer_numpy_array(qapp):
    arr = np.random.randn(180, 360).astype(np.float32)
    viewer = GlobeViewer(arr, cmap="terrain")

    assert viewer.raster_width == 360
    assert viewer.raster_height == 180
    assert viewer.data.shape == (180, 360)
    assert viewer.gl_widget is not None
    assert viewer.title_label.text() == "Interactive Globe Viewer"


def test_globe_viewer_responsive_layout(qapp):
    arr = np.zeros((64, 128), dtype=np.float32)
    viewer = GlobeViewer(arr, display_size=600, responsive=True)

    assert viewer.responsive is True
    assert viewer.gl_widget.minimumWidth() == 200
    assert viewer.gl_widget.minimumHeight() == 200


def test_globe_viewer_alias(qapp):
    arr = np.ones((100, 200), dtype=np.float32)
    viewer = globe_viewer(raster=arr, cmap="viridis")
    assert isinstance(viewer, GlobeViewer)
    assert viewer.raster_width == 200
    assert viewer.raster_height == 100


def test_globe_viewer_with_topo_colormap(qapp):
    arr = np.linspace(-3000, 4000, 180 * 360, dtype=np.float32).reshape(180, 360)
    topo = load_topo_cmap()
    viewer = GlobeViewer(arr, cmap=topo, vmin=-3000, vmax=4000)
    assert viewer.vmin == -3000.0
    assert viewer.vmax == 4000.0


def test_load_topo_cmap_from_resource():
    grayscale = load_topo_cmap(":/legends/grayscale.txt")
    assert grayscale.name == "grayscale"


def test_globe_viewer_pil_image(qapp):
    pil_img = Image.new("F", (360, 180), color=42.0)
    viewer = GlobeViewer(pil_img)
    assert viewer.raster_width == 360
    assert viewer.raster_height == 180
    assert viewer.data[0, 0] == 42.0


def test_globe_viewer_3d_array(qapp):
    arr3d = np.ones((1, 90, 180), dtype=np.float32) * 100.0
    viewer = GlobeViewer(arr3d)
    assert viewer.data.shape == (90, 180)


def test_globe_viewer_geotiff_file(qapp):
    if rasterio is None:
        pytest.skip("rasterio not installed")
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = Path(tmpdir) / "world.tif"
        data = (np.random.rand(180, 360) * 1000).astype(np.float32)
        with rasterio.open(
            tif_path, "w", driver="GTiff", height=180, width=360, count=1, dtype="float32"
        ) as dst:
            dst.write(data, 1)

        with GlobeViewer(tif_path, cmap="terrain") as viewer:
            assert viewer.raster_width == 360
            assert viewer.raster_height == 180
            np.testing.assert_allclose(viewer.data, data, rtol=1e-5)


def test_globe_viewer_npy_and_npz_files(qapp):
    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = Path(tmpdir) / "world.npy"
        npz_path = Path(tmpdir) / "world.npz"
        data = np.arange(180 * 360, dtype=np.float32).reshape(180, 360)

        np.save(npy_path, data)
        np.savez(npz_path, elevation=data)

        v_npy = GlobeViewer(npy_path)
        np.testing.assert_allclose(v_npy.data, data)

        v_npz = GlobeViewer(npz_path)
        np.testing.assert_allclose(v_npz.data, data)


def test_globe_viewer_relief_shading(qapp):
    arr = np.sin(np.linspace(0, 10, 100 * 200, dtype=np.float32)).reshape(100, 200) * 2000.0
    viewer = GlobeViewer(arr, relief=True, relief_intensity=2.0)
    assert viewer.show_relief is True

    viewer.set_relief(False)
    assert viewer.show_relief is False


def test_globe_viewer_set_methods(qapp):
    arr = np.ones((50, 100), dtype=np.float32)
    viewer = GlobeViewer(arr)

    viewer.set_colormap("plasma")
    assert viewer.cmap == "plasma"

    viewer.set_range(-10, 50)
    assert viewer.vmin == -10.0
    assert viewer.vmax == 50.0


def test_globe_viewer_mesh_mode(qapp):
    arr = np.zeros((64, 128), dtype=np.float32)
    viewer = GlobeViewer(arr, mesh=True, mesh_min_level=1, mesh_max_level=3)
    assert viewer.mesh_grid is not None
    assert isinstance(viewer.mesh_grid, IcosphereGrid)
    assert viewer.gl_widget.has_mesh is True
    assert viewer.gl_widget.btn_mesh is not None


def test_globe_viewer_centre_method(qapp):
    arr = np.zeros((64, 128), dtype=np.float32)

    # Default coordinates (0.0, 0.0)
    viewer = GlobeViewer(arr)
    lat, lon = viewer.centre()
    assert lat == 0.0
    assert lon == 0.0
    assert viewer.center() == (0.0, 0.0)

    # Custom initial coordinates via center / centre
    v_custom1 = GlobeViewer(arr, center=(35.5, -120.5))
    assert v_custom1.centre() == (35.5, -120.5)


def test_globe_viewer_missing_source():
    with pytest.raises(ValueError, match="Must provide a raster source"):
        GlobeViewer()
