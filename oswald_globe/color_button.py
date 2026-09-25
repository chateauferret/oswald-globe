"""Reusable color-selection button widget."""

from __future__ import annotations

from typing import Optional, Tuple, Union

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QPushButton, QWidget

from oswald_globe.app_support import color_to_qcolor


class ColorButton(QPushButton):
    def __init__(self, color: Union[QColor, Tuple[float, float, float], str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = color_to_qcolor(color)
        self.clicked.connect(self._choose_color)
        self._update_label()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: Union[QColor, Tuple[float, float, float], str]) -> None:
        self._color = color_to_qcolor(color)
        self._update_label()

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Select color")
        if chosen.isValid():
            self._color = chosen
            self._update_label()

    def _update_label(self) -> None:
        self.setText(self._color.name().upper())
        self.setToolTip(self._color.name().upper())
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; color: white; padding: 4px 8px; }}"
        )
