"""Worker object for building adaptive icosphere meshes."""

from __future__ import annotations

from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from oswald_globe.icosphere import IcosphereBuildCancelled, IcosphereGrid


class IcosphereBuildWorker(QObject):
    progressChanged = Signal(str, int, int)
    completed = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, elevation: np.ndarray, min_level: int, max_level: int, threshold: float):
        super().__init__()
        self._elevation = np.asarray(elevation, dtype=np.float32)
        self._min_level = min_level
        self._max_level = max_level
        self._threshold = threshold
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            grid = IcosphereGrid.from_equirectangular(
                self._elevation,
                min_level=self._min_level,
                max_level=self._max_level,
                threshold=self._threshold,
                progress_callback=self.progressChanged.emit,
                is_cancelled=self._cancel_event.is_set,
            )
        except IcosphereBuildCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(exc)
            return

        self.completed.emit(grid)

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()
