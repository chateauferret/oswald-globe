"""Paint tool configuration and menu action."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QDialog, QWidget

from .tool_options_dialog import ToolOptionsDialog
from .tool import BrushTool


class PaintTool(BrushTool):
    _menu_label = "Paint"

    def __init__(self, parent: QWidget, on_brush_changed: Callable[[], None]):
        super().__init__(parent, on_brush_changed)
        self._value = 0
        self._mode = "Replace"

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = ToolOptionsDialog(
            self._parent,
            value=self._value,
            mode=self._mode,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, ToolOptionsDialog):
            self._value = int(self._options_dialog.value_slider.value())
            self._mode = str(self._options_dialog.mode_combo.currentText())
        super().dispose_options_dialog()

    def value(self) -> int:
        if isinstance(self._options_dialog, ToolOptionsDialog):
            return int(self._options_dialog.value_slider.value())
        return self._value

    def mode(self) -> str:
        if isinstance(self._options_dialog, ToolOptionsDialog):
            return str(self._options_dialog.mode_combo.currentText())
        return self._mode
