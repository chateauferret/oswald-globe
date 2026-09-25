"""Compatibility exports for globe editing undo support."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .brush_paint_command import BrushPaintCommand
    from .undo_history import UndoStack, UndoableCommand
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe.brush_paint_command import BrushPaintCommand
    from oswald_globe.undo_history import UndoStack, UndoableCommand

__all__ = ["BrushPaintCommand", "UndoStack", "UndoableCommand"]
