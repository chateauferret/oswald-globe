"""Shaded relief map generation for terrain data."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from oswald_globe.colormap import load_topo_cmap


def _to_numpy(x: np.ndarray) -> np.ndarray:
    """Convert input to contiguous float32 numpy array."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def get_relief_map(
    elevation: np.ndarray,
    climate: Optional[np.ndarray] = None,
    biome: Optional[np.ndarray] = None,
    flow: Optional[np.ndarray] = None,
    *,
    azimuths: Tuple[float, ...] = (315.0, 45.0, 135.0, 225.0),
    flow_threshold: float = 7.0,
    sigma_large: float = 6.0,
    sigma_small: float = 1.2,
    resolution: float = 90.0,
    rgb: Optional[np.ndarray] = None,
    relief: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Generate GDAL-style multi-scale shaded relief map.

    Args:
        elevation: (H, W) float meters.
        climate: unused placeholder for compatibility.
        biome: unused placeholder for compatibility.
        flow: optional (H, W) flow accumulation for rivers.
        azimuths: illumination azimuths.
        flow_threshold: threshold for river overlay.
        sigma_large: broad feature filter radius.
        sigma_small: local feature filter radius.
        resolution: meters per pixel.
        rgb: optional pre-colorized base image.
        relief: relief intensity multiplier.
        vmin: minimum value for colormap scaling.
        vmax: maximum value for colormap scaling.

    Returns:
        shaded_rgb: (H, W, 3) float32 RGB array normalized to [0, 1].
    """
    elev = _to_numpy(elevation)
    assert elev.ndim == 2, "elevation must be 2D (H, W)"

    azimuth_deg = float(azimuths[0]) if isinstance(azimuths, (tuple, list)) and len(azimuths) > 0 else 315.0
    altitude_deg = 45.0

    elev_f32 = elev.astype(np.float32, copy=False)
    if np.isnan(elev_f32).any():
        median_val = float(np.nanmedian(elev_f32)) if np.isfinite(np.nanmedian(elev_f32)) else 0.0
        elev_f32 = np.nan_to_num(elev_f32, nan=median_val)

    def compute_hillshade(src: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(src)
        scale = max(1e-4, 15.0 * resolution / 90.0)
        dy, dx = dy / scale, dx / scale
        slope_rad = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
        aspect_rad = np.arctan2(dy, -dx)
        az_rad = np.deg2rad(azimuth_deg)
        alt_rad = np.deg2rad(altitude_deg)
        hs = (
            np.sin(alt_rad) * np.sin(slope_rad)
            + np.cos(alt_rad) * np.cos(slope_rad) * np.cos(az_rad - aspect_rad)
        )
        return np.clip(hs, 0.0, 1.0).astype(np.float32)

    elev_large = gaussian_filter(elev_f32, sigma=sigma_large)
    elev_small = gaussian_filter(elev_f32, sigma=sigma_small)

    hs_large = compute_hillshade(elev_large)
    hs_small = compute_hillshade(elev_small)
    hillshade = np.clip(0.75 * hs_large + 0.25 * hs_small, 0.0, 1.0)
    hillshade = np.power(hillshade, 0.85)

    if rgb is None:
        land_elev = np.maximum(0, elev)
        _vmin = float(vmin) if vmin is not None else 0.0
        _vmax = float(vmax) if vmax is not None else 4000.0
        norm = (land_elev - _vmin) / max(1e-6, _vmax - _vmin)
        topo = load_topo_cmap()
        norm_cmap = 0.5 + np.clip(norm ** 0.7, 0.0, 1.0) * 0.5
        base_rgb = topo(norm_cmap)[..., :3].astype(np.float32)
    else:
        base_rgb = np.asarray(rgb, dtype=np.float32)

    intensity = 0.35 + 0.65 * hillshade
    shaded_rgb = np.clip(base_rgb * (relief * intensity + (1.0 - relief))[..., None], 0.0, 1.0)

    # Ocean coloring: fade from light blue (coast) to deep blue
    ocean_mask = elev_f32 < 0.0
    if np.any(ocean_mask):
        depth = -elev_f32
        max_depth = 10_000.0
        t = np.zeros_like(elev_f32, dtype=np.float32)
        t[ocean_mask] = np.clip(depth[ocean_mask] / max_depth, 0.0, 1.0)
        t = t ** 0.7
        t3 = t[..., None]
        coast_color = np.array([0.68, 0.88, 1.00], dtype=np.float32)
        deep_color = np.array([0.00, 0.10, 0.45], dtype=np.float32)
        ocean_rgb = (1.0 - t3) * coast_color + t3 * deep_color
        shaded_rgb = np.where(ocean_mask[..., None], ocean_rgb, shaded_rgb)

    return shaded_rgb
