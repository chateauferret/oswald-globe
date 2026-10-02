"""Tool implementations for the globe application."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, TypeVar

from .paint_tool import PaintTool, PaintToolOptionsDialog
from .select_tool import SelectTool, SelectToolOptionsDialog
from .tool import BrushTool, NavigateTool, Tool

__all__ = [
    "BrushTool",
    "NavigateTool",
    "PaintTool",
    "PaintToolOptionsDialog",
    "SelectTool",
    "SelectToolOptionsDialog",
    "Tool",
    "discover_tools",
    "load_tools",
]

_TOOL_ORDER = {"navigate": 0, "select": 1, "paint": 2}


def _normalize_tool_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized.endswith("_tool"):
        normalized = normalized[:-5]
    return normalized


def _tool_name_for_class(tool_class: type[Tool]) -> str:
    menu_label = getattr(tool_class, "_menu_label", "")
    if menu_label:
        return _normalize_tool_name(str(menu_label))
    class_name = tool_class.__name__
    if class_name.endswith("Tool"):
        return _normalize_tool_name(class_name[:-4])
    return _normalize_tool_name(class_name)


def _instantiate_tool(tool_class: type[Tool], parent: object, on_brush_changed: Callable[[], None]) -> Tool:
    try:
        params = list(inspect.signature(tool_class).parameters)
    except (TypeError, ValueError):
        params = []

    if "on_brush_changed" in params:
        return tool_class(parent, on_brush_changed)
    if "parent" in params:
        return tool_class(parent)
    return tool_class()


def discover_tools() -> dict[str, type[Tool]]:
    discovered: dict[str, type[Tool]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except Exception:
            continue
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate is Tool or candidate is BrushTool:
                continue
            if not issubclass(candidate, Tool):
                continue
            if candidate.__module__ != module.__name__:
                continue
            if getattr(candidate, "__abstractmethods__", None):
                continue
            tool_name = _tool_name_for_class(candidate)
            if not tool_name:
                continue
            discovered.setdefault(tool_name, candidate)
    return discovered


def load_tools(parent: object, on_brush_changed: Callable[[], None]) -> dict[str, Tool]:
    discovered = discover_tools()
    ordered: dict[str, Tool] = {}
    for tool_name in sorted(discovered, key=lambda name: (_TOOL_ORDER.get(name, len(_TOOL_ORDER) + 1), name)):
        tool_class = discovered[tool_name]
        ordered[tool_name] = _instantiate_tool(tool_class, parent, on_brush_changed)
    if not ordered:
        raise RuntimeError(f"No tool plugins were found in {__path__!r}.")
    return ordered
