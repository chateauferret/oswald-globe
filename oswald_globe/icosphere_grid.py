"""Adaptive icosphere mesh for representing global terrain data.

An `IcosphereGrid` is a geodesic sphere built by recursively subdividing an
icosahedron. Unlike a fixed-resolution equirectangular raster, each triangle
can be subdivided independently, so the mesh can be made denser where the
underlying data (e.g. elevation) has steep gradients and left coarse where
it is flat (e.g. oceans, plains) - while still exposing simple vertex/edge
connectivity for grid-based algorithms such as erosion.

Use `IcosphereGrid.from_equirectangular` to build an adaptively refined mesh
from an existing equirectangular raster, and `IcosphereGrid.to_equirectangular`
to rasterize mesh data back onto a regular lat/lon grid.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .layer import Layer, LayerLegend
    from .progress_tracker import ProgressTracker, emit_progress
    from .utils import sample_equirectangular
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe.layer import Layer, LayerLegend
    from oswald_globe.progress_tracker import ProgressTracker, emit_progress
    from oswald_globe.utils import sample_equirectangular

_GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0

_BASE_VERTICES = np.array([
    [-1, _GOLDEN, 0], [1, _GOLDEN, 0], [-1, -_GOLDEN, 0], [1, -_GOLDEN, 0],
    [0, -1, _GOLDEN], [0, 1, _GOLDEN], [0, -1, -_GOLDEN], [0, 1, -_GOLDEN],
    [_GOLDEN, 0, -1], [_GOLDEN, 0, 1], [-_GOLDEN, 0, -1], [-_GOLDEN, 0, 1],
], dtype=np.float64)
_BASE_VERTICES /= np.linalg.norm(_BASE_VERTICES, axis=1, keepdims=True)

_BASE_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


class IcosphereBuildCancelled(Exception):
    """Raised when adaptive icosphere construction is cancelled."""


def _xyz_to_latlon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert unit-sphere xyz coordinates (..., 3) to (lat_deg, lon_deg)."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert (lat_deg, lon_deg) arrays to unit-sphere xyz coordinates, shape (..., 3)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    coslat = np.cos(lat)
    return np.stack([coslat * np.cos(lon), coslat * np.sin(lon), np.sin(lat)], axis=-1)


class _Face:
    """A single triangle in the mesh, possibly subdivided into 4 children."""

    __slots__ = ("v", "level", "children")

    def __init__(self, v: Tuple[int, int, int], level: int):
        self.v = v
        self.level = level
        self.children: Optional[List["_Face"]] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class IcosphereGrid:
    """Adaptive geodesic-sphere mesh for holding global terrain data."""

    SELECTION_LAYER_NAME = "selection_status"

    def __init__(self):
        self._vertices: List[np.ndarray] = [v.copy() for v in _BASE_VERTICES]
        self._layers: "OrderedDict[str, Layer]" = OrderedDict()
        self._ensure_layer("elevation", values=np.full(len(self._vertices), np.nan, dtype=np.float64))
        self._ensure_layer(
            self.SELECTION_LAYER_NAME,
            values=np.zeros(len(self._vertices), dtype=np.float64),
            description="Soft vertex selection mask",
            visible=False,
        )
        self._edge_midpoints: Dict[Tuple[int, int], int] = {}
        self.roots: List[_Face] = [_Face(f, 0) for f in _BASE_FACES]

        self._vertices_arr_cache: Optional[np.ndarray] = None
        self._leaf_cache: Optional[List[_Face]] = None
        self._conforming_cache: Optional[List[_Face]] = None
        self._kdtree: Optional[cKDTree] = None
        self._kdtree_faces: Optional[List[_Face]] = None
        self._dual_cell_segments_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        

    def layer_names(self) -> List[str]:
        """Return the names of the stored data layers in insertion order."""
        return list(self._layers.keys())

    @staticmethod
    def _coerce_legend_reference(legend: Any) -> Optional[LayerLegend]:
        if legend is None:
            return None
        if isinstance(legend, LayerLegend):
            return legend
        if isinstance(legend, str):
            return LayerLegend(name=legend, colormap=None)
        if isinstance(legend, dict):
            if "name" not in legend or "colormap" not in legend:
                raise ValueError("Legend dictionaries must contain both 'name' and 'colormap'.")
            return LayerLegend(name=legend["name"], colormap=legend["colormap"])
        if isinstance(legend, tuple) and len(legend) == 2:
            return LayerLegend(name=legend[0], colormap=legend[1])
        raise TypeError(
            "Legend must be LayerLegend, str, dict(name/colormap), tuple(name, colormap), or None."
        )

    def add_layer(
        self,
        name: str,
        values: Optional[np.ndarray] = None,
        *,
        description: str = "",
        visible: bool = True,
        opacity: float = 100.0,
        legend: Any = None,
        z_index: Optional[int] = None,
    ) -> Layer:
        """Create or replace a layer keyed by name, with one value per vertex."""
        name = str(name)
        if isinstance(values, Layer):
            layer = values
            layer.name = name
            if description:
                layer.description = description
            if visible is not None:
                layer.visible = bool(visible)
            if opacity is not None:
                layer.opacity = float(opacity)
            if legend is not None:
                layer.legend = self._coerce_legend_reference(legend)
            if z_index is not None:
                layer.z_index = int(z_index)
            self._layers[name] = layer
            return layer

        if name in self._layers:
            layer = self._layers[name]
            if values is not None:
                arr = np.asarray(values, dtype=np.float64)
                if arr.shape[0] != len(self._vertices):
                    raise ValueError(f"Layer {name!r} must contain one value per vertex ({len(self._vertices)}), got shape {arr.shape}.")
                layer.values = arr
            if description:
                layer.description = description
            layer.visible = bool(visible)
            layer.opacity = float(opacity)
            if legend is not None:
                layer.legend = self._coerce_legend_reference(legend)
            if z_index is not None:
                layer.z_index = int(z_index)
            return layer

        if values is None:
            values = np.full(
                len(self._vertices),
                self._default_layer_fill_value(name),
                dtype=np.float64,
            )
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != len(self._vertices):
            raise ValueError(f"Layer {name!r} must contain one value per vertex ({len(self._vertices)}), got shape {arr.shape}.")
        layer = Layer(
            name=name,
            values=arr,
            description=description,
            visible=visible,
            opacity=opacity,
            legend=self._coerce_legend_reference(legend),
            z_index=len(self._layers) if z_index is None else int(z_index),
        )
        self._layers[name] = layer
        return layer

    def _ensure_layer(self, name: str, *, values: Optional[np.ndarray] = None, **kwargs: Any) -> Layer:
        """Ensure a layer exists and matches the current vertex count."""
        if name not in self._layers:
            return self.add_layer(name, values, **kwargs)

        layer = self._layers[name]
        if layer.values.shape[0] != len(self._vertices):
            padded = np.full(
                len(self._vertices),
                self._default_layer_fill_value(name),
                dtype=np.float64,
            )
            copy_count = min(layer.values.shape[0], len(self._vertices))
            if copy_count:
                padded[:copy_count] = layer.values[:copy_count]
            layer.values = padded
        return layer

    @classmethod
    def _default_layer_fill_value(cls, name: str) -> float:
        return 0.0 if str(name) == cls.SELECTION_LAYER_NAME else np.nan

    def get_layer(self, name: str) -> Layer:
        """Return the named layer object."""
        return self._ensure_layer(name)

    def set_layer(self, name: str, values: np.ndarray, **kwargs: Any) -> Layer:
        """Set a named data layer for all vertices."""
        return self.add_layer(name, values, **kwargs)

    @property
    def has_selection(self) -> bool:
        selection = self.get_layer(self.SELECTION_LAYER_NAME).values
        return bool(np.any(selection > 0.0))

    @property
    def _values(self) -> np.ndarray:
        """Backward-compatible access to the elevation layer."""
        return self.get_layer("elevation").values

    @_values.setter
    def _values(self, values: np.ndarray) -> None:
        self.set_layer("elevation", values)

    def _invalidate_caches(self):
        self._vertices_arr_cache = None
        self._leaf_cache = None
        self._conforming_cache = None
        self._kdtree = None
        self._kdtree_faces = None
        self._dual_cell_segments_cache = None

    def _add_vertex(self, xyz: np.ndarray) -> int:
        idx = len(self._vertices)
        self._vertices.append(xyz)
        for layer_name in self.layer_names():
            layer = self._layers[layer_name]
            layer.values = np.append(layer.values, self._default_layer_fill_value(layer_name))
        return idx

    @staticmethod
    def _check_cancelled(is_cancelled: Optional[Callable[[], bool]]) -> None:
        if is_cancelled is not None and is_cancelled():
            raise IcosphereBuildCancelled()

    @staticmethod
    def _potential_face_count(level: int, max_level: int) -> int:
        return 4 ** max(0, max_level - level)

    def _midpoint(self, i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        existing = self._edge_midpoints.get(key)
        if existing is not None:
            return existing
        mid = self._vertices[i] + self._vertices[j]
        mid /= np.linalg.norm(mid)
        idx = self._add_vertex(mid)
        self._edge_midpoints[key] = idx
        return idx

    def _subdivide(self, face: _Face):
        v0, v1, v2 = face.v
        a = self._midpoint(v0, v1)
        b = self._midpoint(v1, v2)
        c = self._midpoint(v2, v0)
        face.children = [
            _Face((v0, a, c), face.level + 1),
            _Face((a, v1, b), face.level + 1),
            _Face((c, b, v2), face.level + 1),
            _Face((a, b, c), face.level + 1),
        ]

    def subdivide_uniform(self, levels: int):
        """Uniformly subdivide every current leaf face `levels` times."""
        for _ in range(levels):
            for leaf in self._red_leaves():
                self._subdivide(leaf)
            self._invalidate_caches()

    @property
    def vertices(self) -> np.ndarray:
        """Unit-sphere xyz coordinates of every vertex, shape (N, 3)."""
        if self._vertices_arr_cache is None:
            self._vertices_arr_cache = np.array(self._vertices)
        return self._vertices_arr_cache

    @property
    def values(self) -> np.ndarray:
        """Elevation data assigned to every vertex, shape (N,)."""
        return np.array(self._values)

    def vertex_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lat_deg, lon_deg) arrays, one entry per vertex."""
        return _xyz_to_latlon(self.vertices)

    def leaf_faces(self) -> List[_Face]:
        """Return the current gap-free leaf triangles (see `_conforming_faces`)."""
        return self._conforming_faces()

    def _red_leaves(self) -> List[_Face]:
        """Return the leaves of the red (regular, 4-way) subdivision quadtree."""
        if self._leaf_cache is not None:
            return self._leaf_cache
        leaves: List[_Face] = []
        stack = list(self.roots)
        while stack:
            f = stack.pop()
            if f.is_leaf:
                leaves.append(f)
            else:
                stack.extend(f.children)
        self._leaf_cache = leaves
        return leaves

    def _conforming_faces(self) -> List[_Face]:
        """Resolve red leaves' hanging T-vertices into a gap-free triangulation."""
        if self._conforming_cache is not None:
            return self._conforming_cache

        verts = self.vertices
        faces: List[_Face] = []
        for f in self._red_leaves():
            v0, v1, v2 = f.v
            edges = ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1))
            mids = []
            for a, b, _ in edges:
                key = (a, b) if a < b else (b, a)
                mids.append(self._edge_midpoints.get(key))
            hanging = [i for i, m in enumerate(mids) if m is not None]

            if not hanging:
                faces.append(f)
            elif len(hanging) == 1:
                a, b, opp = edges[hanging[0]]
                m = mids[hanging[0]]
                faces.append(_Face((a, m, opp), f.level))
                faces.append(_Face((m, b, opp), f.level))
            elif len(hanging) == 2:
                i, j = hanging
                a_i, b_i, _ = edges[i]
                a_j, b_j, _ = edges[j]
                m_i, m_j = mids[i], mids[j]
                set_i, set_j = {a_i, b_i}, {a_j, b_j}
                shared = (set_i & set_j).pop()
                far_i = (set_i - {shared}).pop()
                far_j = (set_j - {shared}).pop()
                faces.append(_Face((shared, m_i, m_j), f.level))
                faces.append(_Face((far_i, m_i, m_j), f.level))
                faces.append(_Face((far_i, m_j, far_j), f.level))
            else:
                a01, a12, a20 = mids
                faces.append(_Face((v0, a01, a20), f.level))
                faces.append(_Face((a01, v1, a12), f.level))
                faces.append(_Face((a20, a12, v2), f.level))
                faces.append(_Face((a01, a12, a20), f.level))

        # Guarantee consistent outward-facing winding for rendering (backface culling).
        for i, f in enumerate(faces):
            a, b, c = f.v
            va, vb, vc = verts[a], verts[b], verts[c]
            normal = np.cross(vb - va, vc - va)
            if np.dot(normal, va + vb + vc) < 0:
                faces[i] = _Face((a, c, b), f.level)

        self._conforming_cache = faces
        return faces

    def face_count(self) -> int:
        return len(self.leaf_faces())

    def vertex_count(self) -> int:
        return len(self._vertices)

    def build_adjacency(self) -> Dict[int, Set[int]]:
        """Build undirected vertex adjacency from the current leaf faces."""
        adjacency: Dict[int, Set[int]] = {}
        for f in self.leaf_faces():
            v0, v1, v2 = f.v
            for a, b in ((v0, v1), (v1, v2), (v2, v0)):
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
        return adjacency

    def dual_vertices(self) -> np.ndarray:
        """Return each leaf triangle's centroid, projected onto the unit sphere."""
        faces = self.leaf_faces()
        verts = self.vertices
        tri_indices = np.array([f.v for f in faces])
        centroids = verts[tri_indices].mean(axis=1)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        return centroids

    def dual_edges(self) -> np.ndarray:
        """Return pairs of `dual_vertices()` indices connecting adjacent triangles' centroids."""
        faces = self.leaf_faces()
        edge_to_face: Dict[Tuple[int, int], int] = {}
        edges: List[Tuple[int, int]] = []
        for i, f in enumerate(faces):
            v0, v1, v2 = f.v
            for a, b in ((v0, v1), (v1, v2), (v2, v0)):
                key = (a, b) if a < b else (b, a)
                other = edge_to_face.get(key)
                if other is None:
                    edge_to_face[key] = i
                else:
                    edges.append((other, i))
        return np.array(edges, dtype=np.uint32).reshape(-1, 2)

    def dual_cell_line_segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Voronoi cell line segments and per-vertex draw ranges."""
        if self._dual_cell_segments_cache is not None:
            return self._dual_cell_segments_cache

        faces = self.leaf_faces()
        centroids = self.dual_vertices()
        incident_faces: List[List[int]] = [[] for _ in range(self.vertex_count())]
        for face_index, face in enumerate(faces):
            for vertex_index in face.v:
                incident_faces[vertex_index].append(face_index)

        segment_groups: List[np.ndarray] = []
        ranges = np.zeros((self.vertex_count(), 2), dtype=np.int32)
        vertices = self.vertices
        offset = 0

        for vertex_index, face_indices in enumerate(incident_faces):
            if len(face_indices) < 3:
                ranges[vertex_index] = (offset, 0)
                continue

            center = vertices[vertex_index]
            reference_axis = (
                np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(center[2]) < 0.9
                else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            )
            tangent_x = np.cross(reference_axis, center)
            tangent_x /= np.linalg.norm(tangent_x)
            tangent_y = np.cross(center, tangent_x)

            cell_centroids = centroids[np.asarray(face_indices, dtype=np.intp)]
            projected = cell_centroids - (cell_centroids @ center)[:, None] * center[None, :]
            angles = np.arctan2(projected @ tangent_y, projected @ tangent_x)
            ordered = cell_centroids[np.argsort(angles)]

            pairs = np.empty((len(ordered) * 2, 3), dtype=np.float64)
            pairs[0::2] = ordered
            pairs[1::2] = np.roll(ordered, -1, axis=0)
            segment_groups.append(pairs)
            ranges[vertex_index] = (offset, len(pairs))
            offset += len(pairs)

        if segment_groups:
            segments = np.concatenate(segment_groups, axis=0)
        else:
            segments = np.empty((0, 3), dtype=np.float64)

        self._dual_cell_segments_cache = (segments, ranges)
        return self._dual_cell_segments_cache

    def _ensure_values(self, indices, value_fn):
        missing = [i for i in indices if np.isnan(self._values[i])]
        if not missing:
            return
        xyz = np.array([self._vertices[i] for i in missing])
        lat, lon = _xyz_to_latlon(xyz)
        vals = np.atleast_1d(value_fn(lat, lon))
        for i, v in zip(missing, vals):
            self._values[i] = float(v)

    def _steepness(self, face: _Face) -> float:
        """Estimate |gradient| across a face as max(|dvalue| / angular edge length)."""
        v0, v1, v2 = face.v
        vals = (self._values[v0], self._values[v1], self._values[v2])
        pts = (self._vertices[v0], self._vertices[v1], self._vertices[v2])
        max_grad = 0.0
        for i, j in ((0, 1), (1, 2), (2, 0)):
            dot = np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0)
            angle = np.arccos(dot)
            if angle < 1e-9:
                continue
            grad = abs(vals[i] - vals[j]) / angle
            max_grad = max(max_grad, grad)
        return max_grad

    def _has_unseen_feature(self, face: _Face, value_fn, threshold: float) -> bool:
        """Probe each edge's true midpoint to catch features narrower than the face."""
        v0, v1, v2 = face.v
        pts = (self._vertices[v0], self._vertices[v1], self._vertices[v2])
        vals = (self._values[v0], self._values[v1], self._values[v2])
        edges = ((0, 1), (1, 2), (2, 0))

        mids = []
        half_angles = []
        for i, j in edges:
            dot = np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0)
            half_angles.append(np.arccos(dot) / 2.0)
            mid = pts[i] + pts[j]
            mids.append(mid / np.linalg.norm(mid))

        lat, lon = _xyz_to_latlon(np.array(mids))
        actual = np.atleast_1d(value_fn(lat, lon))
        predicted = np.array([0.5 * (vals[i] + vals[j]) for i, j in edges])
        half_angles = np.array(half_angles)

        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.where(half_angles > 1e-9, np.abs(actual - predicted) / half_angles, 0.0)
        return bool(np.any(grad > threshold))

    def _refine_face(
        self,
        face: _Face,
        value_fn,
        min_level: int,
        max_level: int,
        threshold: float,
        creation_progress: Optional[ProgressTracker] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> _Face:
        self._check_cancelled(is_cancelled)
        self._ensure_values(face.v, value_fn)
        if face.level >= max_level:
            if creation_progress is not None:
                creation_progress.advance(1)
            return face
        should_subdivide = (
            face.level < min_level
            or self._steepness(face) > threshold
            or self._has_unseen_feature(face, value_fn, threshold)
        )
        if should_subdivide:
            if face.children is None:
                self._subdivide(face)
            for k, child in enumerate(face.children):
                face.children[k] = self._refine_face(
                    child,
                    value_fn,
                    min_level,
                    max_level,
                    threshold,
                    creation_progress=creation_progress,
                    is_cancelled=is_cancelled,
                )
        elif creation_progress is not None:
            creation_progress.advance(self._potential_face_count(face.level, max_level))
        return face

    def refine_adaptive(
        self,
        value_fn,
        min_level: int = 4,
        max_level: int = 10,
        threshold: float = 50.0,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        """Recursively subdivide faces where `value_fn` varies steeply."""
        creation_progress = ProgressTracker(
            "creating-faces",
            len(self.roots) * self._potential_face_count(0, max_level),
            progress_callback,
        )
        self.roots = [
            self._refine_face(
                root,
                value_fn,
                min_level,
                max_level,
                threshold,
                creation_progress=creation_progress,
                is_cancelled=is_cancelled,
            )
            for root in self.roots
        ]
        creation_progress.finish()
        self._invalidate_caches()
        self._balance(value_fn, progress_callback=progress_callback, is_cancelled=is_cancelled)
        self._invalidate_caches()
        self._populate_face_values(value_fn, progress_callback=progress_callback, is_cancelled=is_cancelled)
        self._invalidate_caches()

    def _balance(
        self,
        value_fn,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        """Enforce a 2:1 balance."""
        pass_count = 0
        while True:
            self._check_cancelled(is_cancelled)
            pass_count += 1
            emit_progress(progress_callback, "balancing-faces", pass_count - 1, pass_count)
            changed = False
            leaves = self._red_leaves()
            for f in leaves:
                self._check_cancelled(is_cancelled)
                v0, v1, v2 = f.v
                if any(
                    self._far_side_depth(a, b) >= 2
                    for a, b in ((v0, v1), (v1, v2), (v2, v0))
                ):
                    self._subdivide(f)
                    for child in f.children:
                        self._ensure_values(child.v, value_fn)
                    changed = True
            if changed:
                self._invalidate_caches()
                continue
            break
        emit_progress(progress_callback, "balancing-faces", pass_count, pass_count)

    def _populate_face_values(
        self,
        value_fn,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        emit_progress(progress_callback, "populating-faces", 0, 0)
        leaves = self.leaf_faces()
        population_progress = ProgressTracker("populating-faces", len(leaves), progress_callback)
        for face in leaves:
            self._check_cancelled(is_cancelled)
            self._ensure_values(face.v, value_fn)
            population_progress.advance(1)
        population_progress.finish()

    def _far_side_depth(self, a: int, b: int) -> int:
        """How many levels deeper edge (a, b) has been subdivided on the far side."""
        key = (a, b) if a < b else (b, a)
        m = self._edge_midpoints.get(key)
        if m is None:
            return 0
        return 1 + max(self._far_side_depth(a, m), self._far_side_depth(m, b))

    @classmethod
    def from_equirectangular(
        cls,
        arr: np.ndarray,
        min_level: int = 2,
        max_level: int = 8,
        threshold: float = 50.0,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> "IcosphereGrid":
        """Build an adaptive icosphere from a 2D equirectangular raster."""
        grid = cls()
        grid.refine_adaptive(
            lambda lat, lon: sample_equirectangular(arr, lat, lon),
            min_level=min_level,
            max_level=max_level,
            threshold=threshold,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        return grid

    def _build_kdtree(self):
        if self._kdtree is not None:
            return
        leaves = self.leaf_faces()
        verts = self.vertices
        faces_v = np.array([f.v for f in leaves])
        centroids = verts[faces_v].sum(axis=1)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        self._kdtree_faces = leaves
        self._kdtree = cKDTree(centroids)

    def sample(self, lat_deg, lon_deg, k: int = 8) -> np.ndarray:
        """Interpolate mesh values at arbitrary (lat, lon) points."""
        self._build_kdtree()
        leaves = self._kdtree_faces
        k = max(1, min(k, len(leaves)))

        lat_deg = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
        lon_deg = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
        query = _latlon_to_xyz(lat_deg, lon_deg)
        n = query.shape[0]

        _, candidate_idx = self._kdtree.query(query, k=k)
        if k == 1:
            candidate_idx = candidate_idx[:, None]

        verts = self.vertices
        vals = self.values

        result = np.full(n, np.nan)
        resolved = np.zeros(n, dtype=bool)

        for col in range(candidate_idx.shape[1]):
            unresolved = ~resolved
            if not np.any(unresolved):
                break
            idxs = candidate_idx[unresolved, col]
            faces_v = np.array([leaves[i].v for i in idxs])
            A, B, C = verts[faces_v[:, 0]], verts[faces_v[:, 1]], verts[faces_v[:, 2]]
            d = query[unresolved]

            M = np.stack([-d, B - A, C - A], axis=-1)
            rhs = -A
            sol = np.full((M.shape[0], 3), np.nan)
            nonsingular = np.abs(np.linalg.det(M)) > 1e-12
            if np.any(nonsingular):
                sol[nonsingular] = np.linalg.solve(M[nonsingular], rhs[nonsingular, :, None])[..., 0]

            t, u, v = sol[:, 0], sol[:, 1], sol[:, 2]
            w = 1.0 - u - v
            valid = (t > 0) & (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)

            interp = w * vals[faces_v[:, 0]] + u * vals[faces_v[:, 1]] + v * vals[faces_v[:, 2]]

            global_indices = np.where(unresolved)[0]
            newly_resolved = global_indices[valid]
            result[newly_resolved] = interp[valid]
            resolved[newly_resolved] = True

        if not np.all(resolved):
            remaining = np.where(~resolved)[0]
            nearest_leaf = candidate_idx[remaining, 0]
            for r, leaf_i in zip(remaining, nearest_leaf):
                f = leaves[leaf_i]
                result[r] = float(np.mean(vals[list(f.v)]))

        return result

    def to_equirectangular(self, height: int, width: int, k: int = 8) -> np.ndarray:
        """Rasterize the mesh back onto a regular equirectangular grid."""
        rows = np.arange(height)
        cols = np.arange(width)
        lat = 90.0 - (rows + 0.5) / height * 180.0
        lon = (cols + 0.5) / width * 360.0 - 180.0
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        flat_vals = self.sample(lat_grid.ravel(), lon_grid.ravel(), k=k)
        return flat_vals.reshape(height, width)

    def _serialize_face_tree(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        face_vertices: List[Tuple[int, int, int]] = []
        face_levels: List[int] = []
        face_children: List[List[int]] = []

        def visit(face: _Face) -> int:
            face_index = len(face_vertices)
            face_vertices.append(tuple(int(vertex) for vertex in face.v))
            face_levels.append(int(face.level))
            face_children.append([-1, -1, -1, -1])
            if face.children is not None:
                if len(face.children) != 4:
                    raise ValueError("Serialized icosphere faces must have either 0 or 4 children.")
                child_indices = [visit(child) for child in face.children]
                face_children[face_index] = child_indices
            return face_index

        root_indices = [visit(root) for root in self.roots]
        return (
            np.asarray(face_vertices, dtype=np.int64),
            np.asarray(face_levels, dtype=np.int64),
            np.asarray(face_children, dtype=np.int64),
            np.asarray(root_indices, dtype=np.int64),
        )

    def to_serialized_state(self) -> Dict[str, Any]:
        """Return a fully materialized representation of the mesh geometry and layers."""
        face_vertices, face_levels, face_children, root_indices = self._serialize_face_tree()
        edge_items = sorted(self._edge_midpoints.items())
        if edge_items:
            edge_midpoint_keys = np.asarray([key for key, _ in edge_items], dtype=np.int64)
            edge_midpoint_values = np.asarray([value for _, value in edge_items], dtype=np.int64)
        else:
            edge_midpoint_keys = np.empty((0, 2), dtype=np.int64)
            edge_midpoint_values = np.empty((0,), dtype=np.int64)

        layer_metadata = []
        layer_values: Dict[str, np.ndarray] = {}
        for layer_index, layer_name in enumerate(self.layer_names()):
            layer = self._layers[layer_name]
            layer_key = f"layer_values_{layer_index}"
            layer_metadata.append(
                {
                    "name": layer.name,
                    "description": layer.description,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "legend_name": None if layer.legend is None else layer.legend.name,
                    "z_index": layer.z_index,
                    "values_key": layer_key,
                }
            )
            layer_values[layer_key] = np.asarray(layer.values, dtype=np.float64)

        return {
            "vertices": np.asarray(self.vertices, dtype=np.float64),
            "face_vertices": face_vertices,
            "face_levels": face_levels,
            "face_children": face_children,
            "root_indices": root_indices,
            "edge_midpoint_keys": edge_midpoint_keys,
            "edge_midpoint_values": edge_midpoint_values,
            "layer_metadata": layer_metadata,
            **layer_values,
        }

    @classmethod
    def from_serialized_state(cls, state: Mapping[str, object]) -> "IcosphereGrid":
        """Rebuild an icosphere grid from a serialized geometry/layer snapshot."""
        vertices = np.asarray(state["vertices"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Serialized vertices must have shape (N, 3), got {vertices.shape}.")

        face_vertices = np.asarray(state["face_vertices"], dtype=np.int64)
        face_levels = np.asarray(state["face_levels"], dtype=np.int64)
        face_children = np.asarray(state["face_children"], dtype=np.int64)
        root_indices = np.asarray(state["root_indices"], dtype=np.int64)
        if face_vertices.ndim != 2 or face_vertices.shape[1] != 3:
            raise ValueError(
                f"Serialized face vertices must have shape (N, 3), got {face_vertices.shape}."
            )
        if face_levels.shape != (face_vertices.shape[0],):
            raise ValueError(
                "Serialized face levels must contain one entry per serialized face, "
                f"got {face_levels.shape} for {face_vertices.shape[0]} faces."
            )
        if face_children.shape != (face_vertices.shape[0], 4):
            raise ValueError(
                "Serialized face child indices must have shape (N, 4), "
                f"got {face_children.shape}."
            )

        layer_metadata_obj = state.get("layer_metadata")
        if not isinstance(layer_metadata_obj, Sequence):
            raise ValueError("Serialized layer metadata must be a sequence of layer descriptors.")
        layer_metadata = list(layer_metadata_obj)

        grid = cls()
        grid._vertices = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]
        grid._layers = OrderedDict()

        for descriptor in layer_metadata:
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"Invalid serialized layer descriptor: {descriptor!r}")
            layer_name = str(descriptor["name"])
            values_key = str(descriptor["values_key"])
            if values_key not in state:
                raise ValueError(
                    f"Serialized layer {layer_name!r} is missing its value array {values_key!r}."
                )
            legend_name = descriptor.get("legend_name")
            grid.add_layer(
                layer_name,
                np.asarray(state[values_key], dtype=np.float64),
                description=str(descriptor.get("description", "")),
                visible=bool(descriptor.get("visible", True)),
                opacity=float(descriptor.get("opacity", 100.0)),
                legend=None if legend_name is None else str(legend_name),
                z_index=int(descriptor.get("z_index", len(grid._layers))),
            )

        if "elevation" not in grid._layers:
            raise ValueError("Serialized icosphere grid must include an 'elevation' layer.")
        if cls.SELECTION_LAYER_NAME not in grid._layers:
            grid.add_layer(
                cls.SELECTION_LAYER_NAME,
                np.zeros(len(grid._vertices), dtype=np.float64),
                description="Soft vertex selection mask",
                visible=False,
            )
        selection_layer = grid.get_layer(cls.SELECTION_LAYER_NAME)
        selection_layer.values = np.clip(
            np.nan_to_num(selection_layer.values, nan=0.0),
            0.0,
            1.0,
        )

        edge_midpoint_keys = np.asarray(
            state.get("edge_midpoint_keys", np.empty((0, 2), dtype=np.int64)),
            dtype=np.int64,
        )
        edge_midpoint_values = np.asarray(
            state.get("edge_midpoint_values", np.empty((0,), dtype=np.int64)),
            dtype=np.int64,
        )
        if edge_midpoint_keys.size == 0:
            grid._edge_midpoints = {}
        else:
            if edge_midpoint_keys.ndim != 2 or edge_midpoint_keys.shape[1] != 2:
                raise ValueError(
                    "Serialized edge midpoint keys must have shape (N, 2), "
                    f"got {edge_midpoint_keys.shape}."
                )
            if edge_midpoint_values.shape != (edge_midpoint_keys.shape[0],):
                raise ValueError(
                    "Serialized edge midpoint values must contain one entry per edge key, "
                    f"got {edge_midpoint_values.shape} for {edge_midpoint_keys.shape[0]} keys."
                )
            grid._edge_midpoints = {
                (int(key[0]), int(key[1])): int(value)
                for key, value in zip(edge_midpoint_keys, edge_midpoint_values)
            }

        nodes = [
            _Face((int(vertices_row[0]), int(vertices_row[1]), int(vertices_row[2])), int(level))
            for vertices_row, level in zip(face_vertices, face_levels)
        ]
        for node_index, child_indices in enumerate(face_children):
            valid_children = [int(child_index) for child_index in child_indices if int(child_index) >= 0]
            if valid_children:
                if len(valid_children) != 4:
                    raise ValueError(
                        "Serialized non-leaf faces must reference exactly 4 children, "
                        f"got {valid_children!r}."
                    )
                nodes[node_index].children = [nodes[child_index] for child_index in valid_children]
        grid.roots = [nodes[int(root_index)] for root_index in root_indices]
        grid._invalidate_caches()
        return grid
