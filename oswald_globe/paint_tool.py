"""Paint tool configuration and menu action."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QDialog, QMenu, QWidget

from oswald_globe.paint_tool_options_dialog import PaintToolOptionsDialog
from oswald_globe.tool import Tool


class PaintTool(Tool):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._value = 0
        self._mode = "Replace"
        self._radius_km = 100
        self._falloff_percent = 50

    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction("Paint", self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = PaintToolOptionsDialog(
            self._parent,
            value=self._value,
            mode=self._mode,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    @Slot(int)
    def _notify_brush_changed(self, _value: int) -> None:
        sync = getattr(self._parent, "_sync_tool_mode_to_viewer", None)
        if callable(sync):
            sync()

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            self._value = int(self._options_dialog.value_slider.value())
            self._mode = str(self._options_dialog.mode_combo.currentText())
            self._radius_km = int(self._options_dialog.radius_slider.value())
            self._falloff_percent = int(self._options_dialog.falloff_slider.value())
        super().dispose_options_dialog()

    def value(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.value_slider.value())
        return self._value

    def mode(self) -> str:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return str(self._options_dialog.mode_combo.currentText())
        return self._mode

    def radius_km(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.radius_slider.value())
        return self._radius_km

    def falloff_percent(self) -> int:
        if isinstance(self._options_dialog, PaintToolOptionsDialog):
            return int(self._options_dialog.falloff_slider.value())
        return self._falloff_percent
