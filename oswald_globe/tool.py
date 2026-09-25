"""Base tool abstractions for the globe application."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QDialog, QMenu, QWidget


class Tool(ABC):
    def __init__(self, parent: QWidget):
        self._parent = parent
        self._menu_action: Optional[QAction] = None
        self._options_dialog: Optional[QDialog] = None

    @property
    def menu_action(self) -> Optional[QAction]:
        return self._menu_action

    @property
    def options_dialog(self) -> Optional[QDialog]:
        return self._options_dialog

    @abstractmethod
    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        raise NotImplementedError

    def create_options_dialog(self) -> Optional[QDialog]:
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
        dialog.close()
        dialog.deleteLater()


class NavigateTool(Tool):
    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction("Navigate", self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action
