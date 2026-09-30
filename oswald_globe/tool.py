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


class BrushTool(Tool):
    """Common base for tools that manipulate a radius/falloff brush.

    Subclasses provide the menu label (`_menu_label`) and the options
    dialog (`create_options_dialog`), and typically override
    `dispose_options_dialog` to persist any additional fields before
    calling `super().dispose_options_dialog()`.
    """

    _menu_label: str = ""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._radius_km = 100
        self._falloff_percent = 50

    def create_menu_action(
        self,
        tools_menu: QMenu,
        action_group: QActionGroup,
        on_selected: Callable[[], None],
    ) -> QAction:
        action = QAction(self._menu_label, self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action

    @Slot(int)
    def _notify_brush_changed(self, _value: int) -> None:
        sync = getattr(self._parent, "_sync_tool_mode_to_viewer", None)
        if callable(sync):
            sync()

    def dispose_options_dialog(self) -> None:
        if self._options_dialog is not None:
            self._radius_km = int(self._options_dialog.radius_slider.value())
            self._falloff_percent = int(self._options_dialog.falloff_slider.value())
        super().dispose_options_dialog()

    def radius_km(self) -> int:
        if self._options_dialog is not None:
            return int(self._options_dialog.radius_slider.value())
        return self._radius_km

    def falloff_percent(self) -> int:
        if self._options_dialog is not None:
            return int(self._options_dialog.falloff_slider.value())
        return self._falloff_percent
