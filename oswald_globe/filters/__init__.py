"""Filter implementations for the globe application."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from .fill_filter import FillFilter
from .filter import Filter

__all__ = [
    "Filter",
    "FillFilter",
    "discover_filters",
    "load_filters",
]


def _normalize_filter_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized.endswith("_filter"):
        normalized = normalized[:-7]
    return normalized


def _filter_name_for_class(filter_class: type[Filter]) -> str:
    menu_label = getattr(filter_class, "_menu_label", "")
    if menu_label:
        return _normalize_filter_name(str(menu_label))
    class_name = filter_class.__name__
    if class_name.endswith("Filter"):
        return _normalize_filter_name(class_name[:-6])
    return _normalize_filter_name(class_name)


def _instantiate_filter(filter_class: type[Filter], parent: object) -> Filter:
    try:
        params = list(inspect.signature(filter_class).parameters)
    except (TypeError, ValueError):
        params = []

    if "parent" in params:
        return filter_class(parent)
    return filter_class()


def discover_filters() -> dict[str, type[Filter]]:
    discovered: dict[str, type[Filter]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except Exception:
            continue
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate is Filter:
                continue
            if not issubclass(candidate, Filter):
                continue
            if candidate.__module__ != module.__name__:
                continue
            if getattr(candidate, "__abstractmethods__", None):
                continue
            filter_name = _filter_name_for_class(candidate)
            if not filter_name:
                continue
            discovered.setdefault(filter_name, candidate)
    return discovered


def load_filters(parent: object) -> dict[str, Filter]:
    discovered = discover_filters()
    ordered: dict[str, Filter] = {}
    for filter_name in sorted(discovered):
        filter_class = discovered[filter_name]
        ordered[filter_name] = _instantiate_filter(filter_class, parent)
    if not ordered:
        raise RuntimeError(f"No filter plugins were found in {__path__!r}.")
    return ordered
