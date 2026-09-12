"""Geometry, matrix math, coordinate conversions, and array utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def sample_equirectangular(arr: np.ndarray, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
    """Sample values from an equirectangular 2D array using bilinear interpolation.

    The equirectangular map is assumed to cover:
    lat: +90° (row 0) to -90° (row H-1)
    lon: -180° (col 0) to +180° (col W-1)
    """
    h, w = arr.shape
    row = (90.0 - lats_deg) / 180.0 * h - 0.5
    col = (lons_deg + 180.0) / 360.0 * w - 0.5

    r0 = np.floor(row).astype(int)
    r1 = r0 + 1
    dr = row - r0
    r0 = np.clip(r0, 0, h - 1)
    r1 = np.clip(r1, 0, h - 1)

    col_floor = np.floor(col).astype(int)
    dc = col - col_floor
    c0 = col_floor % w
    c1 = (c0 + 1) % w

    return (
        (1.0 - dr) * (1.0 - dc) * arr[r0, c0]
        + (1.0 - dr) * dc * arr[r0, c1]
        + dr * (1.0 - dc) * arr[r1, c0]
        + dr * dc * arr[r1, c1]
    )


def mat4_identity() -> np.ndarray:
    """Return a 4x4 identity matrix as a 16-element float32 array (column-major)."""
    m = np.zeros(16, dtype=np.float32)
    m[0] = 1.0
    m[5] = 1.0
    m[10] = 1.0
    m[15] = 1.0
    return m


def mat4_ortho(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Create an orthographic projection matrix matching WebGL mat4Ortho."""
    lr = 1.0 / (left - right)
    bt = 1.0 / (bottom - top)
    nf = 1.0 / (near - far)
    out = np.zeros(16, dtype=np.float32)
    out[0] = -2.0 * lr
    out[5] = -2.0 * bt
    out[10] = 2.0 * nf
    out[12] = (left + right) * lr
    out[13] = (top + bottom) * bt
    out[14] = (far + near) * nf
    out[15] = 1.0
    return out


def mat4_rotate_x(out: np.ndarray, a: np.ndarray, rad: float) -> np.ndarray:
    """Rotate a 4x4 matrix by `rad` around the X axis in-place or returning new array."""
    s = math.sin(rad)
    c = math.cos(rad)
    a10, a11, a12, a13 = a[4], a[5], a[6], a[7]
    a20, a21, a22, a23 = a[8], a[9], a[10], a[11]

    if a is not out:
        out[0], out[1], out[2], out[3] = a[0], a[1], a[2], a[3]
        out[12], out[13], out[14], out[15] = a[12], a[13], a[14], a[15]

    out[4] = a10 * c + a20 * s
    out[5] = a11 * c + a21 * s
    out[6] = a12 * c + a22 * s
    out[7] = a13 * c + a23 * s
    out[8] = a20 * c - a10 * s
    out[9] = a21 * c - a11 * s
    out[10] = a22 * c - a12 * s
    out[11] = a23 * c - a13 * s
    return out


def mat4_rotate_y(out: np.ndarray, a: np.ndarray, rad: float) -> np.ndarray:
    """Rotate a 4x4 matrix by `rad` around the Y axis in-place or returning new array."""
    s = math.sin(rad)
    c = math.cos(rad)
    a00, a01, a02, a03 = a[0], a[1], a[2], a[3]
    a20, a21, a22, a23 = a[8], a[9], a[10], a[11]

    if a is not out:
        out[4], out[5], out[6], out[7] = a[4], a[5], a[6], a[7]
        out[12], out[13], out[14], out[15] = a[12], a[13], a[14], a[15]

    out[0] = a00 * c - a20 * s
    out[1] = a01 * c - a21 * s
    out[2] = a02 * c - a20 * s  # wait, check standard mat4RotateY
    # Let's verify carefully:
    # out[0] = a00 * c - a20 * s
    # out[1] = a01 * c - a21 * s
    # out[2] = a02 * c - a22 * s
    # out[3] = a03 * c - a23 * s
    # out[8] = a00 * s + a20 * c
    # out[9] = a01 * s + a21 * c
    # out[10] = a02 * s + a22 * c
    # out[11] = a03 * s + a23 * c
    out[2] = a02 * c - a22 * s
    out[3] = a03 * c - a23 * s
    out[8] = a00 * s + a20 * c
    out[9] = a01 * s + a21 * c
    out[10] = a02 * s + a22 * c
    out[11] = a03 * s + a23 * c
    return out


def mat3_normal_from_mat4(a: np.ndarray) -> np.ndarray:
    """Compute the 3x3 normal matrix (inverse transpose) from a 4x4 matrix."""
    a00, a01, a02, a03 = a[0], a[1], a[2], a[3]
    a10, a11, a12, a13 = a[4], a[5], a[6], a[7]
    a20, a21, a22, a23 = a[8], a[9], a[10], a[11]
    a30, a31, a32, a33 = a[12], a[13], a[14], a[15]

    b00 = a00 * a11 - a01 * a10
    b01 = a00 * a12 - a02 * a10
    b02 = a00 * a13 - a03 * a10
    b03 = a01 * a12 - a02 * a11
    b04 = a01 * a13 - a03 * a11
    b05 = a02 * a13 - a03 * a12
    b06 = a20 * a31 - a21 * a30
    b07 = a20 * a32 - a22 * a30
    b08 = a20 * a33 - a23 * a30
    b09 = a21 * a32 - a22 * a31
    b10 = a21 * a33 - a23 * a31
    b11 = a22 * a33 - a23 * a32

    det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
    out = np.zeros(9, dtype=np.float32)
    if abs(det) < 1e-12:
        return out
    inv_det = 1.0 / det

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det
    out[1] = (a12 * b08 - a10 * b11 - a13 * b07) * inv_det
    out[2] = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det
    out[3] = (a02 * b10 - a01 * b11 - a03 * b09) * inv_det
    out[4] = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det
    out[5] = (a01 * b08 - a00 * b10 - a03 * b06) * inv_det
    out[6] = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det
    out[7] = (a32 * b02 - a30 * b05 - a33 * b01) * inv_det
    out[8] = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det
    return out


def get_graticule_step(current_zoom: float) -> float:
    """Select the optimal latitude/longitude grid spacing for current zoom level."""
    view_size = 1.15 / max(0.001, current_zoom)
    visible_span = 2.0 * math.asin(min(1.0, view_size)) * 180.0 / math.pi
    candidate_steps = [
        60.0, 45.0, 30.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0, 1.0,
        0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01,
        0.005, 0.002, 0.001
    ]
    best_step = candidate_steps[0]
    best_diff = 999.0
    for s in candidate_steps:
        count = visible_span / s
        if 5.5 <= count <= 10.5:
            diff = abs(count - 7.5)
            if diff < best_diff:
                best_diff = diff
                best_step = s
    if best_diff < 999.0:
        return best_step
    for s in candidate_steps:
        count = visible_span / s
        diff = abs(count - 7.5)
        if diff < best_diff:
            best_diff = diff
            best_step = s
    return best_step


def unproject_point(
    click_x: float,
    click_y: float,
    width: float,
    height: float,
    zoom: float,
    center_lat: float,
    center_lon: float,
    data_grid: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Unproject 2D canvas coordinates to 3D sphere surface and sample data value."""
    if width <= 0 or height <= 0:
        return None
    if click_x < 0 or click_x > width or click_y < 0 or click_y > height:
        return None

    ndc_x = (click_x / width) * 2.0 - 1.0
    ndc_y = 1.0 - (click_y / height) * 2.0

    aspect = width / height
    view_size = 1.15 / max(0.001, zoom)
    if aspect >= 1.0:
        half_w = view_size * aspect
        half_h = view_size
    else:
        half_w = view_size
        half_h = view_size / aspect

    xv = ndc_x * half_w
    yv = ndc_y * half_h
    r2 = xv * xv + yv * yv
    if r2 > 1.0:
        return None
    zv = math.sqrt(max(0.0, 1.0 - r2))

    # Rotate by -center_lat around X
    cos_lat0 = math.cos(center_lat)
    sin_lat0 = math.sin(center_lat)
    x1 = xv
    y1 = yv * cos_lat0 + zv * sin_lat0
    z1 = -yv * sin_lat0 + zv * cos_lat0

    # Rotate by center_lon around Y
    cos_lon0 = math.cos(center_lon)
    sin_lon0 = math.sin(center_lon)
    x_obj = x1 * cos_lon0 + z1 * sin_lon0
    y_obj = y1
    z_obj = -x1 * sin_lon0 + z1 * cos_lon0

    target_lat = math.asin(max(-1.0, min(1.0, y_obj)))
    target_lon = math.atan2(x_obj, z_obj)

    lat_deg = math.degrees(target_lat)
    lon_deg = (math.degrees(target_lon)) % 360.0
    if lon_deg > 180.0:
        lon_deg -= 360.0
    if lon_deg < -180.0:
        lon_deg += 360.0

    val = None
    if data_grid is not None and data_grid.ndim == 2:
        gh, gw = data_grid.shape
        if gh > 0 and gw > 0:
            v = 0.5 - target_lat / math.pi
            u = target_lon / (2.0 * math.pi) + 0.5
            u = ((u % 1.0) + 1.0) % 1.0
            v = max(0.0, min(1.0, v))
            row = min(gh - 1, max(0, int(round(v * (gh - 1)))))
            col = min(gw - 1, max(0, int(round(u * (gw - 1)))))
            val = float(data_grid[row, col])

    return {
        "mouse_x": click_x,
        "mouse_y": click_y,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "value": val,
    }


CHANNEL_INFO = [
    ("heightmap", 0, 1.0, -1000.0),
    ("temperature", 1, 1.0, None),
    ("temperature_std", 2, 100.0, None),
    ("precipitation", 3, 1.0, None),
    ("precipitation_cv", 4, 1.0, None),
]

CHANNEL_ALIASES = {
    "heightfield": 0,
    "heightmap": 0,
    "elevation": 0,
    "elev": 0,
    "height": 0,
    "dem": 0,
    "temperature": 1,
    "temp": 1,
    "temperature_std": 2,
    "temp_std": 2,
    "temperature_variability": 2,
    "precipitation": 3,
    "precip": 3,
    "rain": 3,
    "precipitation_cv": 4,
    "precip_cv": 4,
    "precipitation_variability": 4,
}


def extract_channels(coarse_maps=None, **kwargs) -> dict[int, np.ndarray]:
    """Extract and normalize channel arrays into a dict mapping channel_idx -> 2D numpy array."""
    channels: dict[int, np.ndarray] = {}

    if coarse_maps is not None:
        if isinstance(coarse_maps, (str, Path)):
            path = Path(coarse_maps)
            if path.suffix == ".npy":
                coarse_maps = np.load(path)
            elif path.suffix == ".npz":
                loaded = np.load(path)
                coarse_maps = {k: loaded[k] for k in loaded.files}
            elif path.is_dir():
                dir_dict = {}
                for f in path.glob("*.npy"):
                    dir_dict[f.stem] = np.load(f)
                coarse_maps = dir_dict

        if isinstance(coarse_maps, np.ndarray):
            if coarse_maps.ndim == 2:
                channels[0] = coarse_maps
            elif coarse_maps.ndim == 3:
                if coarse_maps.shape[0] <= 5 and coarse_maps.shape[0] < coarse_maps.shape[1]:
                    for c in range(coarse_maps.shape[0]):
                        channels[c] = coarse_maps[c]
                elif coarse_maps.shape[2] <= 5:
                    for c in range(coarse_maps.shape[2]):
                        channels[c] = coarse_maps[:, :, c]
                else:
                    raise ValueError(f"Unsupported 3D array shape: {coarse_maps.shape}")
            else:
                raise ValueError(f"Coarse map array must be 2D or 3D, got ndim={coarse_maps.ndim}")
        elif isinstance(coarse_maps, dict):
            for k, v in coarse_maps.items():
                if v is None:
                    continue
                if isinstance(k, int):
                    channels[k] = np.asarray(v)
                elif isinstance(k, str):
                    k_lower = k.lower()
                    if k_lower in CHANNEL_ALIASES:
                        channels[CHANNEL_ALIASES[k_lower]] = np.asarray(v)
                    else:
                        try:
                            ch_idx = int(k)
                            channels[ch_idx] = np.asarray(v)
                        except ValueError:
                            raise ValueError(f"Unknown channel name '{k}'.")
        elif isinstance(coarse_maps, (list, tuple)):
            for c, v in enumerate(coarse_maps):
                if v is not None:
                    channels[c] = np.asarray(v)

    for k, v in kwargs.items():
        if v is None:
            continue
        k_lower = k.lower()
        if k_lower in CHANNEL_ALIASES:
            channels[CHANNEL_ALIASES[k_lower]] = np.asarray(v)

    if not channels:
        raise ValueError("No coarse map data provided.")

    shapes = {}
    for ch, arr in channels.items():
        if arr.ndim != 2:
            raise ValueError(f"Channel {ch} array must be 2D, got shape {arr.shape}")
        shapes[ch] = arr.shape

    first_shape = next(iter(shapes.values()))
    for ch, shape in shapes.items():
        if shape != first_shape:
            raise ValueError(f"All coarse map channel arrays must have matching 2D shapes. Got shapes: {shapes}")

    return channels
