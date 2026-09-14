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

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

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


class _ProgressTracker:
    def __init__(self, phase: str, total: int, callback: Optional[Callable[[str, int, int], None]]):
        self.phase = phase
        self.total = max(1, int(total))
        self.callback = callback
        self.done = 0
        self._last_emitted = -1
        self._emit_every = max(1, self.total // 512)
        self._emit()

    def advance(self, amount: int) -> None:
        self.done = min(self.total, self.done + int(amount))
        if self.done == self.total or self.done - self._last_emitted >= self._emit_every:
            self._emit()

    def finish(self) -> None:
        self.done = self.total
        self._emit()

    def _emit(self) -> None:
        if self.callback is None or self.done == self._last_emitted:
            return
        self.callback(self.phase, self.done, self.total)
        self._last_emitted = self.done


def _emit_progress(
    callback: Optional[Callable[[str, int, int], None]],
    phase: str,
    done: int,
    total: int,
) -> None:
    if callback is not None:
        callback(phase, done, total)


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

    def __init__(self):
        self._vertices: List[np.ndarray] = [v.copy() for v in _BASE_VERTICES]
        self._values: List[float] = [np.nan] * len(self._vertices)
        self._edge_midpoints: Dict[Tuple[int, int], int] = {}
        self.roots: List[_Face] = [_Face(f, 0) for f in _BASE_FACES]

        self._vertices_arr_cache: Optional[np.ndarray] = None
        self._leaf_cache: Optional[List[_Face]] = None
        self._conforming_cache: Optional[List[_Face]] = None
        self._kdtree: Optional[cKDTree] = None
        self._kdtree_faces: Optional[List[_Face]] = None

    def _invalidate_caches(self):
        self._vertices_arr_cache = None
        self._leaf_cache = None
        self._conforming_cache = None
        self._kdtree = None
        self._kdtree_faces = None

    def _add_vertex(self, xyz: np.ndarray) -> int:
        idx = len(self._vertices)
        self._vertices.append(xyz)
        self._values.append(np.nan)
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
        """Data value assigned to every vertex, shape (N,)."""
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
        creation_progress: Optional[_ProgressTracker] = None,
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
        creation_progress = _ProgressTracker(
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
            _emit_progress(progress_callback, "balancing-faces", pass_count - 1, pass_count)
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
        _emit_progress(progress_callback, "balancing-faces", pass_count, pass_count)

    def _populate_face_values(
        self,
        value_fn,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        _emit_progress(progress_callback, "populating-faces", 0, 0)
        leaves = self.leaf_faces()
        population_progress = _ProgressTracker("populating-faces", len(leaves), progress_callback)
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
