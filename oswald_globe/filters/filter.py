"""Base filter abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget

from oswald_globe.tools.tool_options_dialog import ToolOptionsDialog


class Filter(ABC):
    def __init__(self, parent: QWidget):
        self._parent = parent
        self._menu_action: Optional[QAction] = None
        self._options_dialog: Optional[ToolOptionsDialog] = None

    @property
    def menu_action(self) -> Optional[QAction]:
        return self._menu_action

    @property
    def options_dialog(self) -> Optional[ToolOptionsDialog]:
        return self._options_dialog

    @abstractmethod
    def create_menu_action(
        self,
        filters_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        raise NotImplementedError

    def create_options_dialog(self) -> Optional[ToolOptionsDialog]:
        return None

    def show_options_dialog(self) -> None:
        if self._options_dialog is None:
            self._options_dialog = self.create_options_dialog()
            if self._options_dialog is not None:
                self._options_dialog.destroyed.connect(self._on_options_dialog_destroyed)
        if self._options_dialog is None:
            return
        self._options_dialog.show()
        self._options_dialog.raise_()
        self._options_dialog.activateWindow()

    @Slot()
    def _on_options_dialog_destroyed(self) -> None:
        self._options_dialog = None

    def dispose_options_dialog(self) -> None:
        if self._options_dialog is None:
            return
        dialog = self._options_dialog
        self._options_dialog = None
        if isinstance(self._parent, QMainWindow):
            self._parent.removeDockWidget(dialog)
        dialog.close()
        dialog.deleteLater()

    @abstractmethod
    def apply(self, gl_widget: Any) -> None:
        raise NotImplementedError
