"""Undo stack and placeholder brush commands for globe editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol


class UndoableCommand(Protocol):
    def redo(self) -> None: ...

    def undo(self) -> None: ...


class UndoStack:
    """Simple undo/redo stack that executes commands on push."""

    def __init__(self, on_changed: Optional[Callable[[], None]] = None):
        self._commands: List[UndoableCommand] = []
        self._index = 0
        self._on_changed = on_changed

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._commands)

    def push(self, command: UndoableCommand) -> None:
        del self._commands[self._index :]
        self._commands.append(command)
        self._index = len(self._commands)
        command.redo()
        self._notify_changed()

    def undo(self) -> None:
        if not self.can_undo():
            return
        self._index -= 1
        self._commands[self._index].undo()
        self._notify_changed()

    def redo(self) -> None:
        if not self.can_redo():
            return
        command = self._commands[self._index]
        self._index += 1
        command.redo()
        self._notify_changed()

    def _notify_changed(self) -> None:
        if self._on_changed is not None:
            self._on_changed()


@dataclass(slots=True)
class BrushPaintCommand:
    """Placeholder brush command used to wire undo/redo before painting exists."""

    target_lat_deg: float
    target_lon_deg: float
    radius_km: float
    falloff_percent: float
    paint_value: int
    paint_mode: str

    def redo(self) -> None:
        print(
            "[brush] apply paint stroke placeholder "
            f"(lat={self.target_lat_deg:.4f}, lon={self.target_lon_deg:.4f}, "
            f"radius_km={self.radius_km:.1f}, falloff={self.falloff_percent:.1f}%, "
            f"mode={self.paint_mode!r}, value={self.paint_value})"
        )
        print("  TODO: apply the brush stroke to the globe data and refresh the texture.")

    def undo(self) -> None:
        print(
            "[brush] undo paint stroke placeholder "
            f"(lat={self.target_lat_deg:.4f}, lon={self.target_lon_deg:.4f}, "
            f"radius_km={self.radius_km:.1f}, falloff={self.falloff_percent:.1f}%, "
            f"mode={self.paint_mode!r}, value={self.paint_value})"
        )
        print("  TODO: restore the previous globe data and refresh the texture.")
