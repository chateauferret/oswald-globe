# Filter plugins for Oswald Globe

This project supports a plugin system for filters. The application does not hardcode the filter list in the main window. Instead, at startup it scans the `oswald_globe/filters` package for Python modules and loads any concrete `Filter` subclasses it finds.

Filters are whole-selection operations. They are applied over the current mesh selection, with each vertex weighted by the selection value (`0.0` to `1.0`). If there is no selection, the effective selection is treated as "all vertices selected".

If you add a new file to `oswald_globe/filters`, the app will discover it automatically when the application starts.

## Where to put a plugin

Create a new Python file in:

`oswald_globe/filters/`

For example:

`oswald_globe/filters/my_filter.py`

The file may contain:

- one `Filter` subclass
- one or more helper classes
- one optional Qt options dialog class
- one optional undoable command class

The file name can be anything you like, as long as it does not start with `_`.

## What the application discovers

The loader looks for any class that:

- is a subclass of `Filter`
- is not the abstract base `Filter` itself
- is defined in the module being imported
- is not abstract

It normalizes the filter name from either the `_menu_label` or the class name, and then creates a menu action for it under the app's `Filters` menu.

## Required base classes

The common base classes live in `oswald_globe/filters/filter.py`.

### `Filter`

Every filter plugin should inherit from `Filter`.

Required methods:

- `create_menu_action(self, filters_menu, action_group, on_selected) -> QAction`
  - Creates the menu entry in the Filters menu.
  - Must add the action to `filters_menu` and `action_group`.
  - Usually sets `self._menu_action = action`.

- `apply(self, gl_widget) -> None`
  - Executes the filter on the current globe mesh.
  - This is the main operation entry point for the plugin.

Optional methods:

- `create_options_dialog(self) -> Optional[QDialog]`
  - Returns a Qt dialog for filter settings.
  - Called when the filter is selected from the menu and the app wants to show its parameter dialog.

- `show_options_dialog(self)`
  - Built into the base class; normally you do not override it.

- `dispose_options_dialog(self)`
  - Built into the base class; normally you do not override it.

## Selection semantics

Filters operate on the whole selection rather than a local brush area.

The effective rule is:

- if some vertices are selected, apply the filter to those vertices only
- if no vertices are selected, treat the entire globe as selected
- selection values are used as weights during the operation

This is how the built-in `FillFilter` works: it computes an operation over all vertices but blends each result by the corresponding selection weight.

## Minimum example: a simple filter

```python
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from oswald_globe.filters.filter import Filter


class MyFilter(Filter):
    _menu_label = "My Filter"

    def __init__(self, parent=None):
        super().__init__(parent)

    def create_menu_action(self, filters_menu: QMenu, action_group: QActionGroup, on_selected):
        action = QAction(self._menu_label, self._parent)
        action.triggered.connect(lambda checked=False: on_selected())
        filters_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action

    def apply(self, gl_widget):
        # Execute the filter here.
        return None
```

This is enough for the app to discover the filter and add it to the Filters menu.

## Example: a parameterized filter

This is the pattern used by the built-in `FillFilter`.

```python
from typing import Any, Optional

from PySide6.QtWidgets import QDialog, QWidget

from oswald_globe.filters.filter import Filter


class MyFilterOptionsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("My Filter Options")
        self.resize(360, 180)


class MyFilter(Filter):
    _menu_label = "My Filter"

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._value = 0
        self._mode = "Replace"

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = MyFilterOptionsDialog(self._parent)
        return dialog

    def apply(self, gl_widget: Any | None = None) -> None:
        if gl_widget is None:
            return

        mesh_grid = gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Filter requires a mesh-backed globe.")

        # Compute from mesh_grid and push an undoable command.
        # The operation is usually implemented as a command object with redo()/undo().
        pass
```

## Creating a custom options dialog

Filter settings should usually live in the same file as the filter class. This keeps the plugin self-contained and mirrors the built-in pattern used by the Fill filter.

A dialog can subclass `QDialog` directly, or subclass the shared `ToolOptionsDialog` from `oswald_globe/tools/tool_options_dialog.py` if you want to reuse the standard slider/spinbox pattern.

Typical dialog responsibilities:

- expose filter parameters
- store the current values on the filter instance
- apply the operation immediately when the user chooses Apply
- commit the edit when the user clicks OK
- close or keep the dialog open depending on the user action

The built-in Fill filter exposes a value slider and mode selector, and it provides the expected `Apply` / `OK` / `Cancel` behavior.

## Undoable commands and execution flow

Filter logic is typically implemented as an undoable command, which is pushed onto the app’s `UndoStack`.

A command should support:

- `redo()`
- `undo()`

This matches the model used elsewhere in the app: an operation computes the new mesh state, writes it to the underlying layer, and can be undone later.

For the Fill filter, the command computes the full result for all vertices (or selected vertices) and applies it as one atomic edit.

## Important rules for plugin files

- Put the new file in `oswald_globe/filters/`
- Use a concrete subclass of `Filter`
- Do not rely on the main app importing the class explicitly
- The application discovers it automatically from the package directory
- Keep the filter dialog and filter logic in the same file when possible

## Example directory layout

```text
oswald_globe/
  filters/
    __init__.py
    filter.py
    fill_filter.py
    my_custom_filter.py
```

The `my_custom_filter.py` file can define:

- a `Filter` subclass
- a Qt parameter dialog class
- an undoable command class for redo/undo
- optional helper utilities

## Built-in example: Fill filter

The built-in `FillFilter` demonstrates the intended architecture:

- it is discovered by scanning `oswald_globe/filters`
- it adds a menu action under `Filters`
- it shows a parameter dialog with `Apply`, `OK`, and `Cancel`
- it applies the selected operation over the whole selection or the whole globe
- it stores its work as an undoable command so it can be undone/redone

If you want to create a custom global filter, the `FillFilter` is the best starting point to copy and modify.

## Summary

Filter plugins in Oswald Globe are intentionally simple:

- discover automatically from `oswald_globe/filters`
- subclass `Filter`
- add a menu action with `create_menu_action()`
- optionally show a parameter dialog
- apply the selected operation in `apply()`
- push undoable commands when the edit should be reversible

This keeps the application generic while allowing custom filter logic to be written as self-contained, drop-in modules.
