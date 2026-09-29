"""Compatibility entrypoint for the desktop application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_nvidia_egl_vendor_library_file() -> str | None:
    candidate_dirs = (
        Path("/usr/share/glvnd/egl_vendor.d"),
        Path("/usr/local/share/glvnd/egl_vendor.d"),
        Path("/etc/glvnd/egl_vendor.d"),
    )
    for candidate_dir in candidate_dirs:
        if not candidate_dir.is_dir():
            continue
        for candidate in sorted(candidate_dir.glob("*nvidia*.json")):
            if candidate.is_file():
                return str(candidate)
    return None


def _configure_qt_runtime_environment() -> None:
    os.environ.setdefault("QT_OPENGL", "desktop")
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_glx")
        if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
            vendor_library = _find_nvidia_egl_vendor_library_file()
            if vendor_library is not None:
                os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor_library


_configure_qt_runtime_environment()

try:
    from .globe_main_window import (
        GlobeMainWindow,
        IcosphereBuildWorker,
        IcosphereProgressDialog,
        create_window,
        load_elevation,
        main,
    )
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe.globe_main_window import (
        GlobeMainWindow,
        IcosphereBuildWorker,
        IcosphereProgressDialog,
        create_window,
        load_elevation,
        main,
    )

__all__ = [
    "GlobeMainWindow",
    "IcosphereBuildWorker",
    "IcosphereProgressDialog",
    "create_window",
    "load_elevation",
    "main",
]


if __name__ == "__main__":
    main()
