"""PySide6 / PyOpenGL Interactive Globe Viewer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    import rasterio
except ImportError:
    rasterio = None

from oswald_globe.colormap import load_topo_cmap
from oswald_globe.globe_widget import GlobeGLWidget
from oswald_globe.icosphere import IcosphereGrid

STYLE_FILE = Path(__file__).resolve().parent / "styles" / "style.qss"
DEFAULT_COLORMAP_VMIN = -32767.0
DEFAULT_COLORMAP_VMAX = 32767.0


class GlobeViewer(QWidget):
    """Interactive 3D Globe Viewer desktop widget for equirectangular rasters.

    Accepts a 2D equirectangular raster (file path, numpy array, or PIL Image)
    together with a colormap, and constructs an interactive 3D desktop globe using
    PySide6 and PyOpenGL that can be rotated, panned, and zoomed.
    """

    def __init__(
        self,
        raster: Union[str, Path, np.ndarray, Image.Image, None] = None,
        cmap: Union[str, Any, None] = "terrain",
        display_size: int = 700,
        width: Optional[int] = None,
        height: Optional[int] = None,
        responsive: bool = False,
        vmin: float = DEFAULT_COLORMAP_VMIN,
        vmax: float = DEFAULT_COLORMAP_VMAX,
        max_texture_size: int = 2048,
        relief: bool = False,
        relief_intensity: float = 1.5,
        resolution: float = 90.0,
        auto_rotate: bool = False,
        lighting: bool = True,
        title: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        data: Optional[np.ndarray] = None,
        array: Optional[np.ndarray] = None,
        source: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        source_image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        coarse_map: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        centre: Optional[Tuple[float, float]] = None,
        graticule: bool = True,
        mesh: Union[bool, Any, None] = False,
        mesh_min_level: int = 1,
        mesh_max_level: int = 7,
        mesh_threshold: float = 400.0,
        mesh_wireframe: bool = True,
        parent: Optional[QWidget] = None,
        **kwargs: Any,
    ):
        super().__init__(parent)
        self.setObjectName("GlobeContainer")

        self.custom_width = width or display_size
        self.custom_height = height or display_size
        self.responsive = bool(responsive)
        self.cmap = cmap or "terrain"
        self.max_texture_size = max_texture_size
        self.show_relief = bool(relief)
        self.relief_intensity = float(relief_intensity)
        self.resolution = float(resolution)
        self.auto_rotate = bool(auto_rotate)
        self.lighting = bool(lighting)
        self.show_graticule = bool(kwargs.get("show_graticule", graticule))
        self.title = title or "Interactive Globe Viewer"
        self.vmin = float(vmin) if vmin is not None else DEFAULT_COLORMAP_VMIN
        self.vmax = float(vmax) if vmax is not None else DEFAULT_COLORMAP_VMAX

        initial_center = centre if centre is not None else center
        if initial_center is not None:
            self.lat = float(initial_center[0])
            self.lon = float(initial_center[1])
        else:
            self.lat = float(lat) if lat is not None else 0.0
            self.lon = float(lon) if lon is not None else 0.0

        # Resolve raster source
        src = (
            raster
            if raster is not None
            else (
                path
                if path is not None
                else (
                    data
                    if data is not None
                    else (
                        array
                        if array is not None
                        else (
                            source
                            if source is not None
                            else (
                                source_image
                                if source_image is not None
                                else (coarse_map if coarse_map is not None else kwargs.get("coarse_maps"))
                            )
                        )
                    )
                )
            )
        )

        if src is None:
            raise ValueError("Must provide a raster source (file path or numpy array) to GlobeViewer.")

        self._ds = None
        self.path = None

        if isinstance(src, (str, Path)):
            self.path = str(src)
            self._load_from_path(self.path)
        elif isinstance(src, Image.Image):
            arr = np.asarray(src, dtype=np.float32)
            self._load_from_array(arr)
        elif isinstance(src, np.ndarray):
            self._load_from_array(src)
        else:
            arr = np.asarray(src, dtype=np.float32)
            self._load_from_array(arr)

        self.mesh_grid = None
        self.mesh_wireframe = bool(mesh_wireframe)
        if mesh:
            if isinstance(mesh, IcosphereGrid):
                self.mesh_grid = mesh
            else:
                self.mesh_grid = IcosphereGrid.from_equirectangular(
                    self.data, min_level=mesh_min_level, max_level=mesh_max_level, threshold=mesh_threshold
                )

        self._load_stylesheet()
        self._build_ui()

    def _load_stylesheet(self):
        if STYLE_FILE.is_file():
            with STYLE_FILE.open("r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _load_from_array(self, arr: np.ndarray) -> None:
        data = np.asarray(arr, dtype=np.float32)
        if data.ndim == 3:
            if data.shape[0] <= 5:
                data = data[0]
            elif data.shape[2] <= 5:
                data = data[:, :, 0]
            else:
                data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for globe raster, got shape {data.shape}")

        self.data = data
        self.raster_height, self.raster_width = self.data.shape

    def _load_from_path(self, file_path: str) -> None:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".tif", ".tiff") and rasterio is not None:
            try:
                self._ds = rasterio.open(path)
                data = self._ds.read(1, out_dtype=np.float32)
                self.data = data
                self.raster_width = self._ds.width
                self.raster_height = self._ds.height
                return
            except Exception:
                pass

        if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            img = Image.open(path)
            self._load_from_array(np.asarray(img, dtype=np.float32))
        elif suffix == ".npy":
            loaded = np.load(path)
            self._load_from_array(loaded)
        elif suffix == ".npz":
            loaded = np.load(path)
            key = loaded.files[0]
            self._load_from_array(loaded[key])
        else:
            try:
                img = Image.open(path)
                self._load_from_array(np.asarray(img, dtype=np.float32))
            except Exception:
                loaded = np.load(path)
                self._load_from_array(loaded)

    def _get_colormap(self):
        if isinstance(self.cmap, str):
            if self.cmap.lower() == "topo":
                return load_topo_cmap()
            from oswald_globe.colormap import get_named_cmap

            return get_named_cmap(self.cmap)
        return self.cmap

    def _colorize(self, data: np.ndarray) -> np.ndarray:
        cm = self._get_colormap()
        vmin, vmax = float(self.vmin), float(self.vmax)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)
        rgba = cm(norm)
        return rgba[:, :, :3].astype(np.float32)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Header Row
        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.title, self)
        self.title_label.setObjectName("TitleLabel")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.status_label = QLabel(
            f"Lat: {self.lat:.1f}° | Lon: {self.lon:.1f}° | Zoom: 1.00x", self
        )
        self.status_label.setObjectName("StatusLabel")
        header_layout.addWidget(self.status_label)
        main_layout.addLayout(header_layout)

        # Center OpenGL Viewport
        self.gl_widget = GlobeGLWidget(
            data=self.data,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            max_texture_size=self.max_texture_size,
            relief=self.show_relief,
            relief_intensity=self.relief_intensity,
            resolution=self.resolution,
            auto_rotate=self.auto_rotate,
            lighting=self.lighting,
            graticule=self.show_graticule,
            graticule_color=(0.9, 0.95, 1.0),
            mesh_grid=self.mesh_grid,
            mesh_wireframe=self.mesh_wireframe,
            mesh_color=(0.05, 0.05, 0.05),
            lat=self.lat,
            lon=self.lon,
            parent=self,
        )
        self.gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if not self.responsive:
            self.gl_widget.setMinimumSize(max(100, self.custom_width - 24), max(100, self.custom_height - 70))
        else:
            self.gl_widget.setMinimumSize(200, 200)

        self.gl_widget.statusChanged.connect(self._on_status_changed)
        self.gl_widget.centerChanged.connect(self._on_center_changed)
        main_layout.addWidget(self.gl_widget, stretch=1)

        # Footer Row
        footer_layout = QHBoxLayout()
        if self.mesh_grid is not None:
            bottom_left_str = f"{self.mesh_grid.vertex_count():,} vertices"
        else:
            bottom_left_str = "🖱️ Drag: Rotate | Scroll: Zoom"

        self.footer_left_label = QLabel(bottom_left_str, self)
        self.footer_left_label.setObjectName("FooterLabel")
        footer_layout.addWidget(self.footer_left_label)

        footer_layout.addStretch()

        cmap_name = getattr(self.cmap, "name", str(self.cmap))
        self.footer_right_label = QLabel(
            f"Raster: {self.raster_width}×{self.raster_height} | Colormap: {cmap_name}", self
        )
        self.footer_right_label.setObjectName("FooterLabel")
        footer_layout.addWidget(self.footer_right_label)
        main_layout.addLayout(footer_layout)

        if not self.responsive:
            self.resize(self.custom_width, self.custom_height)

    def _on_status_changed(self, lat_deg: float, lon_deg: float, zoom: float):
        self.lat = lat_deg
        self.lon = lon_deg
        self.status_label.setText(f"Lat: {lat_deg:.1f}° | Lon: {lon_deg:.1f}° | Zoom: {zoom:.2f}x")

    def _on_center_changed(self, lat_deg: float, lon_deg: float):
        self.lat = lat_deg
        self.lon = lon_deg

    def center(self) -> Tuple[float, float]:
        """Return current center coordinates (lat, lon) in degrees."""
        return (self.lat, self.lon)

    def centre(self) -> Tuple[float, float]:
        """Alias for center()."""
        return self.center()

    def set_colormap(self, cmap: Union[str, Colormap]) -> GlobeViewer:
        """Update colormap and redraw."""
        self.cmap = cmap
        self.gl_widget.cmap = cmap
        cmap_name = getattr(cmap, "name", str(cmap))
        self.footer_right_label.setText(
            f"Raster: {self.raster_width}×{self.raster_height} | Colormap: {cmap_name}"
        )
        self.gl_widget.update_texture()
        return self

    def set_relief(self, enabled: bool, intensity: Optional[float] = None) -> GlobeViewer:
        """Enable or disable relief shading."""
        self.show_relief = bool(enabled)
        self.gl_widget.show_relief = self.show_relief
        if intensity is not None:
            self.relief_intensity = float(intensity)
            self.gl_widget.relief_intensity = self.relief_intensity
        self.gl_widget.update_texture()
        return self

    def set_range(self, vmin: float, vmax: float) -> GlobeViewer:
        """Set elevation range for colormap mapping."""
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.gl_widget.vmin = self.vmin
        self.gl_widget.vmax = self.vmax
        self.gl_widget.update_texture()
        return self

    def set_graticule(
        self,
        enabled: bool,
        color: Optional[Union[QColor, Tuple[float, float, float], str]] = None,
        opacity: Optional[float] = None,
    ) -> GlobeViewer:
        """Set graticule visibility and optional color."""
        self.show_graticule = bool(enabled)
        self.gl_widget.set_graticule_settings(self.show_graticule, color, opacity)
        return self

    def set_mesh_wireframe(
        self,
        enabled: bool,
        color: Optional[Union[QColor, Tuple[float, float, float], str]] = None,
        opacity: Optional[float] = None,
    ) -> GlobeViewer:
        """Set icosphere wireframe visibility and optional color."""
        self.mesh_wireframe = bool(enabled)
        self.gl_widget.set_mesh_wireframe_settings(self.mesh_wireframe, color, opacity)
        return self

    def print_stats(self) -> None:
        """Print summary statistics for the underlying raster data."""
        finite = self.data[np.isfinite(self.data)]
        if len(finite) > 0:
            name = self.path if self.path is not None else "NumPy Array"
            print(f"Statistics for {name}:")
            print(f"  Shape:  {self.data.shape}")
            print(f"  Min:    {np.min(finite):.2f}")
            print(f"  Max:    {np.max(finite):.2f}")
            print(f"  Mean:   {np.mean(finite):.2f}")
            print(f"  Median: {np.median(finite):.2f}")
            print(f"  Std:    {np.std(finite):.2f}")

    def __enter__(self) -> GlobeViewer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


globe_viewer = GlobeViewer
