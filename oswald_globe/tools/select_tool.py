"""Selection tool configuration and menu action."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QDialog, QWidget

from .tool_options_dialog import ToolOptionsDialog
from .tool import BrushTool


class SelectTool(BrushTool):
    _menu_label = "Select"

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = ToolOptionsDialog(
            self._parent,
            title="Select Tool Options",
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
            include_value=False,
            include_mode=False,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    def __init__(self, parent: QWidget, on_brush_changed: Callable[[], None]):
        super().__init__(parent, on_brush_changed)
