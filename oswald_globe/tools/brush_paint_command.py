"""Compatibility exports for brush-based mesh edit commands."""

from __future__ import annotations

from .brush_command import BrushCommand


def __getattr__(name: str):
    if name == "BrushPaintCommand":
        from .paint_tool import BrushPaintCommand

        return BrushPaintCommand
    if name == "SelectionBrushCommand":
        from .select_tool import SelectionBrushCommand

        return SelectionBrushCommand
    if name == "SelectionLayerCommand":
        from .select_tool import SelectionLayerCommand

        return SelectionLayerCommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BrushCommand", "BrushPaintCommand", "SelectionBrushCommand", "SelectionLayerCommand"]
