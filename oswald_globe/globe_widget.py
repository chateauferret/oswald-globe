"""PyOpenGL / PySide6 Interactive 3D Globe Widget."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as raw_glVertexAttribPointer
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QToolTip,
    QWidget,
)

from oswald_globe.colormap import load_topo_cmap
from oswald_globe.icosphere import IcosphereGrid
from oswald_globe.relief_map import get_relief_map
from oswald_globe.utils import (
    get_graticule_step,
    mat3_normal_from_mat4,
    mat4_identity,
    mat4_ortho,
    mat4_rotate_x,
    mat4_rotate_y,
    unproject_point,
)

SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


def _color_to_rgb(color: Union[QColor, Tuple[float, float, float], Tuple[float, float, float, float], str]) -> Tuple[float, float, float]:
    if isinstance(color, QColor):
        return (color.redF(), color.greenF(), color.blueF())
    if isinstance(color, str):
        qcolor = QColor(color)
        if not qcolor.isValid():
            raise ValueError(f"Invalid color value: {color!r}")
        return (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
    return (float(color[0]), float(color[1]), float(color[2]))


def _load_shader_source(filename: str) -> str:
    path = SHADERS_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        src = f.read()
    if not src.startswith("#version"):
        # Use GLSL 120 for desktop OpenGL compatibility
        src = "#version 120\n" + src
    return src


def _compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not status:
        log = glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        glDeleteShader(shader)
        type_str = "Vertex" if shader_type == GL_VERTEX_SHADER else "Fragment"
        raise RuntimeError(f"{type_str} shader compilation failed:\n{log}\nSource:\n{source}")
    return shader


def _create_program(vs_source: str, fs_source: str) -> int:
    vs = _compile_shader(vs_source, GL_VERTEX_SHADER)
    fs = _compile_shader(fs_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    status = glGetProgramiv(program, GL_LINK_STATUS)
    if not status:
        log = glGetProgramInfoLog(program)
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        glDeleteProgram(program)
        raise RuntimeError(f"Shader program linking failed:\n{log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


class GlobeGLWidget(QOpenGLWidget):
    """OpenGL 3D Globe Viewport Widget."""

    statusChanged = Signal(float, float, float)  # lat_deg, lon_deg, zoom
    centerChanged = Signal(float, float)        # lat_deg, lon_deg

    def __init__(
        self,
        data: np.ndarray,
        cmap: Any = "terrain",
        vmin: float = -4000.0,
        vmax: float = 4000.0,
        max_texture_size: int = 2048,
        relief: bool = False,
        relief_intensity: float = 1.5,
        resolution: float = 90.0,
        auto_rotate: bool = False,
        lighting: bool = True,
        graticule: bool = True,
        graticule_color: Union[QColor, Tuple[float, float, float], Tuple[float, float, float, float], str] = (0.9, 0.95, 1.0),
        graticule_opacity: float = 1.0,
        mesh_grid: Optional[IcosphereGrid] = None,
        mesh_wireframe: bool = True,
        mesh_color: Union[QColor, Tuple[float, float, float], Tuple[float, float, float, float], str] = (0.05, 0.05, 0.05),
        mesh_opacity: float = 1.0,
        lat: float = 0.0,
        lon: float = 0.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.raw_data = data
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.max_texture_size = max_texture_size
        self.show_relief = bool(relief)
        self.relief_intensity = float(relief_intensity)
        self.resolution = float(resolution)
        self.lighting = bool(lighting)
        self.is_graticule = bool(graticule)
        self.graticule_color = _color_to_rgb(graticule_color)
        self.graticule_opacity = float(graticule_opacity)
        self.mesh_grid = mesh_grid
        self.has_mesh = mesh_grid is not None
        self.show_wireframe = bool(mesh_wireframe)
        self.mesh_color = _color_to_rgb(mesh_color)
        self.mesh_opacity = float(mesh_opacity)

        self.center_lat = math.radians(float(lat))
        self.center_lon = math.radians(float(lon))
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 25.0

        self.is_auto_spin = bool(auto_rotate)
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.vel_x = 0.0
        self.vel_y = 0.0

        # Downsample data grid for tooltips and texture if needed
        self.data_grid = self._prepare_grid_data(self.raw_data)

        # GL Resource handles
        self.program = 0
        self.line_program = 0
        self.texture_id = 0
        self.color_lut_texture_id = 0

        self.pos_vbo = 0
        self.norm_vbo = 0
        self.tex_vbo = 0
        self.height_vbo = 0
        self.index_ebo = 0
        self.dual_pos_vbo = 0
        self.dual_line_ebo = 0

        self.index_count = 0
        self.dual_line_count = 0
        self._gl_initialized = False

        # Animation timer (approx 60 FPS)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start()

        # Build UI overlay controls
        self._init_overlay_ui()

    def _prepare_grid_data(self, data: np.ndarray) -> np.ndarray:
        h, w = data.shape
        max_dim = max(h, w)
        if max_dim > self.max_texture_size:
            scale = self.max_texture_size / float(max_dim)
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            from PIL import Image
            pil_temp = Image.fromarray(data)
            resized = pil_temp.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            return np.asarray(resized, dtype=np.float32)
        return np.ascontiguousarray(data, dtype=np.float32)

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

    def _generate_texture_image(self) -> np.ndarray:
        grid = self.data_grid
        rgb = self._colorize(grid)
        if self.show_relief:
            try:
                shaded = get_relief_map(
                    grid,
                    None,
                    None,
                    None,
                    resolution=self.resolution,
                    relief=self.relief_intensity,
                    vmin=self.vmin,
                    vmax=self.vmax,
                    rgb=rgb,
                )
                img_data = (np.clip(shaded, 0.0, 1.0) * 255.0).astype(np.uint8)
            except Exception:
                img_data = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_data = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(img_data)

    def _generate_lut_image(self) -> np.ndarray:
        lut_values = np.linspace(self.vmin, self.vmax, 256, dtype=np.float32).reshape(1, -1)
        lut_rgb = np.clip(self._colorize(lut_values)[0], 0.0, 1.0)
        lut_data = (lut_rgb * 255.0).astype(np.uint8).reshape(1, 256, 3)
        return np.ascontiguousarray(lut_data)

    # ---------------- UI Overlay Controls ----------------

    def _init_overlay_ui(self):
        # Controls Container
        self.controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

        self.btn_zoom_in = QPushButton("+", self.controls_widget)
        self.btn_zoom_in.setProperty("class", "GlobeControlBtn")
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.clicked.connect(self.zoom_in)

        self.btn_zoom_out = QPushButton("−", self.controls_widget)
        self.btn_zoom_out.setProperty("class", "GlobeControlBtn")
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)

        self.btn_reset = QPushButton("⟲", self.controls_widget)
        self.btn_reset.setProperty("class", "GlobeControlBtn")
        self.btn_reset.setToolTip("Reset View")
        self.btn_reset.clicked.connect(self.reset_view)

        self.btn_spin = QPushButton("▶" if not self.is_auto_spin else "⏸", self.controls_widget)
        self.btn_spin.setCheckable(True)
        self.btn_spin.setChecked(self.is_auto_spin)
        self.btn_spin.setProperty("class", "GlobeControlBtn")
        self.btn_spin.setToolTip("Toggle Auto-Spin")
        self.btn_spin.clicked.connect(self.toggle_spin)

        self.btn_grid = QPushButton("🌐", self.controls_widget)
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(self.is_graticule)
        self.btn_grid.setProperty("class", "GlobeControlBtn")
        self.btn_grid.setToolTip("Toggle Graticule")
        self.btn_grid.clicked.connect(self.toggle_graticule)

        controls_layout.addWidget(self.btn_zoom_in)
        controls_layout.addWidget(self.btn_zoom_out)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.btn_spin)
        controls_layout.addWidget(self.btn_grid)

        if self.has_mesh:
            self.btn_mesh = QPushButton("⬡", self.controls_widget)
            self.btn_mesh.setCheckable(True)
            self.btn_mesh.setChecked(self.show_wireframe)
            self.btn_mesh.setProperty("class", "GlobeControlBtn")
            self.btn_mesh.setToolTip("Toggle Icosphere Dual (Voronoi) Mesh")
            self.btn_mesh.clicked.connect(self.toggle_mesh_wireframe)
            controls_layout.addWidget(self.btn_mesh)
        else:
            self.btn_mesh = None

        self.controls_widget.adjustSize()

    def zoom_in(self):
        self.zoom = min(self.max_zoom, self.zoom * 1.33)
        self._emit_status()
        self.update()

    def zoom_out(self):
        self.zoom = max(self.min_zoom, self.zoom / 1.33)
        self._emit_status()
        self.update()

    def reset_view(self):
        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 1.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self._emit_status()
        self._emit_center()
        self.update()

    def toggle_spin(self):
        self.is_auto_spin = not self.is_auto_spin
        self.btn_spin.setChecked(self.is_auto_spin)
        self.btn_spin.setText("⏸" if self.is_auto_spin else "▶")
        self.update()

    def toggle_graticule(self):
        self.set_graticule_settings(not self.is_graticule)

    def toggle_mesh_wireframe(self):
        self.set_mesh_wireframe_settings(not self.show_wireframe)

    def set_graticule_settings(
        self,
        visible: bool,
        color: Union[QColor, Tuple[float, float, float], Tuple[float, float, float, float], str, None] = None,
        opacity: Optional[float] = None,
    ):
        self.is_graticule = bool(visible)
        if color is not None:
            self.graticule_color = _color_to_rgb(color)
        if opacity is not None:
            self.graticule_opacity = max(0.0, min(1.0, float(opacity)))
        self.btn_grid.setChecked(self.is_graticule)
        self.update()

    def set_mesh_wireframe_settings(
        self,
        visible: bool,
        color: Union[QColor, Tuple[float, float, float], Tuple[float, float, float, float], str, None] = None,
        opacity: Optional[float] = None,
    ):
        self.show_wireframe = bool(visible)
        if color is not None:
            self.mesh_color = _color_to_rgb(color)
        if opacity is not None:
            self.mesh_opacity = max(0.0, min(1.0, float(opacity)))
        if self.btn_mesh:
            self.btn_mesh.setChecked(self.show_wireframe)
        self.update()

    def _emit_status(self):
        lat_deg = math.degrees(self.center_lat)
        lon_deg = (math.degrees(self.center_lon)) % 360.0
        if lon_deg > 180.0:
            lon_deg -= 360.0
        if lon_deg < -180.0:
            lon_deg += 360.0
        self.statusChanged.emit(lat_deg, lon_deg, self.zoom)

    def _emit_center(self):
        lat_deg = math.degrees(self.center_lat)
        lon_deg = (math.degrees(self.center_lon)) % 360.0
        if lon_deg > 180.0:
            lon_deg -= 360.0
        if lon_deg < -180.0:
            lon_deg += 360.0
        self.centerChanged.emit(lat_deg, lon_deg)

    # ---------------- OpenGL Implementation ----------------

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.04, 0.05, 0.08, 1.0)

        vs_src = _load_shader_source("globe.vert")
        fs_src = _load_shader_source("globe.frag")
        self.program = _create_program(vs_src, fs_src)

        line_vs_src = _load_shader_source("wireframe.vert")
        line_fs_src = _load_shader_source("wireframe.frag")
        self.line_program = _create_program(line_vs_src, line_fs_src)

        self._init_geometry()
        self._init_textures()
        self._gl_initialized = True

    def _init_geometry(self):
        if self.has_mesh and self.mesh_grid is not None:
            grid = self.mesh_grid
            positions = grid.vertices[:, [1, 2, 0]].astype(np.float32)
            normals = positions
            heights = grid.values.astype(np.float32)
            tex_coords = np.zeros((len(positions), 2), dtype=np.float32)

            leaves = grid.leaf_faces()
            indices = np.array([f.v for f in leaves], dtype=np.uint32).reshape(-1)

            dual_positions = grid.dual_vertices()[:, [1, 2, 0]].astype(np.float32)
            dual_line_indices = grid.dual_edges().astype(np.uint32).reshape(-1)
        else:
            lat_bands = 64
            lon_bands = 64
            radius = 1.0
            pos_list = []
            norm_list = []
            tex_list = []
            idx_list = []

            for i in range(lat_bands + 1):
                v = i / lat_bands
                lat = (0.5 - v) * math.pi
                sin_lat = math.sin(lat)
                cos_lat = math.cos(lat)

                for j in range(lon_bands + 1):
                    u = j / lon_bands
                    lon = (u - 0.5) * 2.0 * math.pi
                    sin_lon = math.sin(lon)
                    cos_lon = math.cos(lon)

                    x = cos_lat * sin_lon
                    y = sin_lat
                    z = cos_lat * cos_lon

                    pos_list.extend([radius * x, radius * y, radius * z])
                    norm_list.extend([x, y, z])
                    tex_list.extend([u, v])

            for lat in range(lat_bands):
                for lon in range(lon_bands):
                    first = lat * (lon_bands + 1) + lon
                    second = first + lon_bands + 1
                    idx_list.extend([first, second, first + 1])
                    idx_list.extend([first + 1, second, second + 1])

            positions = np.array(pos_list, dtype=np.float32)
            normals = np.array(norm_list, dtype=np.float32)
            tex_coords = np.array(tex_list, dtype=np.float32)
            heights = np.zeros(len(pos_list) // 3, dtype=np.float32)
            indices = np.array(idx_list, dtype=np.uint32)
            dual_positions = None
            dual_line_indices = None

        self.index_count = len(indices)

        # Buffers
        self.pos_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)

        self.norm_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_vbo)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        self.tex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.tex_vbo)
        glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)

        self.height_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.height_vbo)
        glBufferData(GL_ARRAY_BUFFER, heights.nbytes, heights, GL_STATIC_DRAW)

        self.index_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        if dual_positions is not None and dual_line_indices is not None and len(dual_line_indices) > 0:
            self.dual_line_count = len(dual_line_indices)
            self.dual_pos_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.dual_pos_vbo)
            glBufferData(GL_ARRAY_BUFFER, dual_positions.nbytes, dual_positions, GL_STATIC_DRAW)

            self.dual_line_ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.dual_line_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, dual_line_indices.nbytes, dual_line_indices, GL_STATIC_DRAW)
        else:
            self.dual_line_count = 0

    def _init_textures(self):
        img_data = self._generate_texture_image()
        h, w, _ = img_data.shape

        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        lut_data = self._generate_lut_image()
        self.color_lut_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_lut_texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, lut_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def update_texture(self):
        if not self._gl_initialized:
            return
        self.makeCurrent()
        img_data = self._generate_texture_image()
        h, w, _ = img_data.shape
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        lut_data = self._generate_lut_image()
        glBindTexture(GL_TEXTURE_2D, self.color_lut_texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, lut_data)
        self.doneCurrent()
        self.update()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        if hasattr(self, "controls_widget"):
            cw = self.controls_widget.width()
            ch = self.controls_widget.height()
            self.controls_widget.move(w - cw - 10, h - ch - 10)

    def paintGL(self):
        if not self._gl_initialized or self.program <= 0 or self.line_program <= 0:
            return

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / float(h)
        view_size = 1.15 / max(0.001, self.zoom)

        if aspect >= 1.0:
            left = -view_size * aspect
            right = view_size * aspect
            bottom = -view_size
            top = view_size
        else:
            left = -view_size
            right = view_size
            bottom = -view_size / aspect
            top = view_size / aspect

        p_matrix = mat4_ortho(left, right, bottom, top, -10.0, 10.0)

        mv_matrix = mat4_identity()
        mv_matrix = mat4_rotate_x(mv_matrix, mv_matrix, self.center_lat)
        mv_matrix = mat4_rotate_y(mv_matrix, mv_matrix, -self.center_lon)

        n_matrix = mat3_normal_from_mat4(mv_matrix)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)

        # Uniform locations
        u_p_matrix = glGetUniformLocation(self.program, "uPMatrix")
        u_mv_matrix = glGetUniformLocation(self.program, "uMVMatrix")
        u_n_matrix = glGetUniformLocation(self.program, "uNMatrix")
        u_sampler = glGetUniformLocation(self.program, "uSampler")
        u_color_lut = glGetUniformLocation(self.program, "uColorLut")
        u_lighting = glGetUniformLocation(self.program, "uLighting")
        u_atmosphere = glGetUniformLocation(self.program, "uAtmosphere")
        u_graticule = glGetUniformLocation(self.program, "uGraticule")
        u_graticule_step = glGetUniformLocation(self.program, "uGraticuleStep")
        u_graticule_opacity = glGetUniformLocation(self.program, "uGraticuleOpacity")
        u_pixel_size_deg = glGetUniformLocation(self.program, "uPixelSizeDeg")
        u_use_vertex_color = glGetUniformLocation(self.program, "uUseVertexColor")
        u_vmin = glGetUniformLocation(self.program, "uVmin")
        u_vmax = glGetUniformLocation(self.program, "uVmax")

        glUniformMatrix4fv(u_p_matrix, 1, GL_FALSE, p_matrix)
        glUniformMatrix4fv(u_mv_matrix, 1, GL_FALSE, mv_matrix)
        glUniformMatrix3fv(u_n_matrix, 1, GL_FALSE, n_matrix)

        glUniform1f(u_lighting, 1.0 if self.lighting else 0.0)
        glUniform1f(u_atmosphere, 0.8)

        grat_step = get_graticule_step(self.zoom)
        visible_span_deg = 2.0 * math.asin(min(1.0, view_size)) * 180.0 / math.pi
        pixel_size_deg = visible_span_deg / max(1.0, float(h))

        glUniform1f(u_graticule, 1.0 if self.is_graticule else 0.0)
        glUniform1f(u_graticule_step, grat_step)
        glUniform1f(u_graticule_opacity, self.graticule_opacity)
        glUniform1f(u_pixel_size_deg, pixel_size_deg)
        glUniform1f(u_use_vertex_color, 1.0 if self.has_mesh else 0.0)
        glUniform1f(u_vmin, self.vmin)
        glUniform1f(u_vmax, self.vmax)
        glUniform3f(glGetUniformLocation(self.program, "uGraticuleColor"), *self.graticule_color)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(u_sampler, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.color_lut_texture_id)
        glUniform1i(u_color_lut, 1)

        # Attribute bindings
        a_position = glGetAttribLocation(self.program, "aPosition")
        a_normal = glGetAttribLocation(self.program, "aNormal")
        a_tex_coord = glGetAttribLocation(self.program, "aTexCoord")
        a_height = glGetAttribLocation(self.program, "aHeight")

        if a_position >= 0:
            glEnableVertexAttribArray(a_position)
            glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
            raw_glVertexAttribPointer(a_position, 3, GL_FLOAT, GL_FALSE, 0, None)

        if a_normal >= 0:
            glEnableVertexAttribArray(a_normal)
            glBindBuffer(GL_ARRAY_BUFFER, self.norm_vbo)
            raw_glVertexAttribPointer(a_normal, 3, GL_FLOAT, GL_FALSE, 0, None)

        if a_tex_coord >= 0:
            glEnableVertexAttribArray(a_tex_coord)
            glBindBuffer(GL_ARRAY_BUFFER, self.tex_vbo)
            raw_glVertexAttribPointer(a_tex_coord, 2, GL_FLOAT, GL_FALSE, 0, None)

        if a_height >= 0:
            glEnableVertexAttribArray(a_height)
            glBindBuffer(GL_ARRAY_BUFFER, self.height_vbo)
            raw_glVertexAttribPointer(a_height, 1, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_ebo)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        # Draw wireframe overlay if enabled
        if self.has_mesh and self.show_wireframe and self.dual_line_count > 0:
            glUseProgram(self.line_program)
            line_u_p_matrix = glGetUniformLocation(self.line_program, "uPMatrix")
            line_u_mv_matrix = glGetUniformLocation(self.line_program, "uMVMatrix")
            line_u_color = glGetUniformLocation(self.line_program, "uWireframeColor")
            line_u_opacity = glGetUniformLocation(self.line_program, "uWireframeOpacity")
            line_a_position = glGetAttribLocation(self.line_program, "aPosition")

            glUniformMatrix4fv(line_u_p_matrix, 1, GL_FALSE, p_matrix)
            glUniformMatrix4fv(line_u_mv_matrix, 1, GL_FALSE, mv_matrix)
            glUniform3f(line_u_color, *self.mesh_color)
            glUniform1f(line_u_opacity, self.mesh_opacity)

            glDepthMask(GL_FALSE)
            glDepthFunc(GL_LEQUAL)
            if line_a_position >= 0:
                glEnableVertexAttribArray(line_a_position)
                glBindBuffer(GL_ARRAY_BUFFER, self.dual_pos_vbo)
                raw_glVertexAttribPointer(line_a_position, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.dual_line_ebo)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDrawElements(GL_LINES, self.dual_line_count, GL_UNSIGNED_INT, None)
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)
            glDepthFunc(GL_LESS)

    # ---------------- Mouse & Animation Handlers ----------------

    def _on_anim_tick(self):
        needs_redraw = False
        if self.is_auto_spin and not self.is_dragging:
            self.center_lon += 0.005 / max(1.0, self.zoom * 0.5)
            self._emit_status()
            needs_redraw = True
        elif not self.is_dragging and (abs(self.vel_x) > 0.0001 or abs(self.vel_y) > 0.0001):
            self.center_lon -= self.vel_x
            self.center_lat += self.vel_y
            self.center_lat = max(-math.pi / 2, min(math.pi / 2, self.center_lat))
            self.vel_x *= 0.92
            self.vel_y *= 0.92
            self._emit_status()
            needs_redraw = True

        if needs_redraw:
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            QToolTip.hideText()
            self.last_mouse_x = event.position().x()
            self.last_mouse_y = event.position().y()
            self.vel_x = 0.0
            self.vel_y = 0.0

    def mouseMoveEvent(self, event: QMouseEvent):
        x = event.position().x()
        y = event.position().y()

        if self.is_dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            self.last_mouse_x = x
            self.last_mouse_y = y

            speed = math.pi / max(1.0, float(self.height()) * self.zoom)
            self.vel_x = dx * speed
            self.vel_y = dy * speed
            self.center_lon -= self.vel_x
            self.center_lat += self.vel_y
            self.center_lat = max(-math.pi / 2, min(math.pi / 2, self.center_lat))

            self._emit_status()
            self.update()
        else:
            self._update_tooltip(x, y)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._emit_center()
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()
            pt = unproject_point(
                x, y, self.width(), self.height(), self.zoom, self.center_lat, self.center_lon, self.data_grid
            )
            if pt is not None:
                self.center_lat = pt["target_lat"]
                self.center_lon = pt["target_lon"]
                self.vel_x = 0.0
                self.vel_y = 0.0
                self._emit_status()
                self._emit_center()
                self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        zoom_factor = math.exp(delta * 0.0015)
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * zoom_factor))
        self._emit_status()
        self.update()
        self._update_tooltip(event.position().x(), event.position().y())

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def _update_tooltip(self, client_x: float, client_y: float):
        if self.is_dragging:
            QToolTip.hideText()
            return

        pt = unproject_point(
            client_x, client_y, self.width(), self.height(), self.zoom, self.center_lat, self.center_lon, self.data_grid
        )
        if pt is None:
            QToolTip.hideText()
            return

        val_text = "N/A"
        if pt["value"] is not None and np.isfinite(pt["value"]):
            v = pt["value"]
            if abs(v) >= 1000 or int(v) == v:
                val_text = f"{v:.1f}"
            elif abs(v) < 0.01 and v != 0:
                val_text = f"{v:.2e}"
            else:
                val_text = f"{v:.2f}"

        tip_x = int(client_x + 14)
        tip_y = int(client_y + 14)

        text = f"Lat: {pt['lat_deg']:.1f}° | Lon: {pt['lon_deg']:.1f}° | Value: {val_text}"
        tip_pos = self.mapToGlobal(QPoint(tip_x, tip_y))
        QToolTip.showText(tip_pos, text, self)
