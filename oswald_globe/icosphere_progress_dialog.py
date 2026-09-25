"""Progress dialog for asynchronous icosphere building."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget


class IcosphereProgressDialog(QDialog):
    cancelRequested = Signal()
    _STAGES = (
        ("loading-heightfield", "Loading heightfield file"),
        ("creating-faces", "Creating icosphere faces"),
        ("balancing-faces", "Balancing icosphere faces"),
        ("populating-faces", "Populating globe data"),
        ("displaying-globe", "Displaying globe"),
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Loading Heightfield")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._stage_labels: Dict[str, QLabel] = {}
        self._stage_bars: Dict[str, QProgressBar] = {}

        layout = QVBoxLayout(self)
        for phase, title in self._STAGES:
            row_layout = QVBoxLayout()
            title_layout = QHBoxLayout()

            title_label = QLabel(title, self)
            state_label = QLabel("Pending", self)
            title_layout.addWidget(title_label)
            title_layout.addStretch()
            title_layout.addWidget(state_label)

            progress_bar = QProgressBar(self)
            progress_bar.setRange(0, 1)
            progress_bar.setValue(0)
            progress_bar.setFormat("Pending")

            row_layout.addLayout(title_layout)
            row_layout.addWidget(progress_bar)
            layout.addLayout(row_layout)

            self._stage_labels[phase] = state_label
            self._stage_bars[phase] = progress_bar

        buttons = QDialogButtonBox(self)
        self._cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel_button.clicked.connect(self.cancel)
        layout.addWidget(buttons)

    def cancel(self) -> None:
        self.cancelRequested.emit()

    def update_progress(self, phase: str, done: int, total: int) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        if total <= 0:
            progress_bar.setRange(0, 0)
            progress_bar.setFormat("Working...")
            state_label.setText("Running")
            return
        maximum = max(1, int(total))
        progress_bar.setRange(0, maximum)
        progress_bar.setValue(min(int(done), maximum))
        progress_bar.setFormat(f"{done:,} / {maximum:,}")
        if done >= maximum:
            state_label.setText("Done")
        elif done > 0:
            state_label.setText("Running")
        else:
            state_label.setText("Starting")

    def start_stage(self, phase: str, total: int = 1) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        maximum = max(1, int(total))
        progress_bar.setRange(0, maximum)
        progress_bar.setValue(0)
        progress_bar.setFormat(f"0 / {maximum:,}")
        state_label.setText("Running")

    def finish_stage(self, phase: str, total: int = 1) -> None:
        self.update_progress(phase, total, total)

    def set_stage_busy(self, phase: str) -> None:
        progress_bar = self._stage_bars[phase]
        state_label = self._stage_labels[phase]
        progress_bar.setRange(0, 0)
        progress_bar.setFormat("Working...")
        state_label.setText("Running")

    def show_cancelling(self) -> None:
        self._cancel_button.setEnabled(False)
        self.setWindowTitle("Cancelling Heightfield Load")
