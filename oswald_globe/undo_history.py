"""Undo stack implementation for globe editing commands."""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol


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
