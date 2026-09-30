"""Select tool configuration and menu action."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog

from oswald_globe.paint_tool_options_dialog import PaintToolOptionsDialog
from oswald_globe.tool import BrushTool


class SelectTool(BrushTool):
    _menu_label = "Select"

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = PaintToolOptionsDialog(
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
