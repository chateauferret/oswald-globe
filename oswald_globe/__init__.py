"""Oswald Globe: Interactive 3D desktop globe viewer in PySide6 and PyOpenGL."""

from importlib import import_module

__version__ = "0.1.0"

__all__ = ["GlobeViewer", "globe_viewer", "IcosphereGrid", "load_topo_cmap"]


def __getattr__(name: str):
    if name == "load_topo_cmap":
        return import_module("oswald_globe.colormap").load_topo_cmap
    if name in {"GlobeViewer", "globe_viewer"}:
        module = import_module("oswald_globe.globe_viewer")
        return getattr(module, name)
    if name == "IcosphereGrid":
        return import_module("oswald_globe.icosphere").IcosphereGrid
    raise AttributeError(f"module 'oswald_globe' has no attribute {name!r}")
