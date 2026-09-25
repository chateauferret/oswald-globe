"""Settings dialog for the globe application."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QVBoxLayout, QWidget

from oswald_globe.globe_viewer import GlobeViewer
from oswald_globe.grid_settings_tab import GridSettingsTab


class SettingsDialog(QDialog):
    def __init__(
        self,
        viewer: GlobeViewer,
        apply_callback: Callable[[Dict[str, Dict[str, Union[bool, float, QColor]]]], None],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 220)
        self._apply_callback = apply_callback

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.grid_tab = GridSettingsTab(viewer, self.tabs)
        self.tabs.addTab(self.grid_tab, "Grids")
        main_layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(self)
        self.apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self.ok_button = buttons.addButton("OK", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_button.clicked.connect(self.apply_changes)
        self.ok_button.clicked.connect(self.accept_changes)
        self.cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(buttons)

    def apply_changes(self) -> None:
        self._apply_callback(self.grid_tab.settings())

    def accept_changes(self) -> None:
        self.apply_changes()
        self.accept()
