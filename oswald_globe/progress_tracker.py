"""Progress helpers for adaptive icosphere generation."""

from __future__ import annotations

from typing import Callable, Optional


class ProgressTracker:
    def __init__(self, phase: str, total: int, callback: Optional[Callable[[str, int, int], None]]):
        self.phase = phase
        self.total = max(1, int(total))
        self.callback = callback
        self.done = 0
        self._last_emitted = -1
        self._emit_every = max(1, self.total // 512)
        self._emit()

    def advance(self, amount: int) -> None:
        self.done = min(self.total, self.done + int(amount))
        if self.done == self.total or self.done - self._last_emitted >= self._emit_every:
            self._emit()

    def finish(self) -> None:
        self.done = self.total
        self._emit()

    def _emit(self) -> None:
        if self.callback is None or self.done == self._last_emitted:
            return
        self.callback(self.phase, self.done, self.total)
        self._last_emitted = self.done


def emit_progress(
    callback: Optional[Callable[[str, int, int], None]],
    phase: str,
    done: int,
    total: int,
) -> None:
    if callback is not None:
        callback(phase, done, total)
