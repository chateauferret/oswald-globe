# Tool plugins for Oswald Globe

This project supports a simple plugin system for tools. The application does not hardcode the tool list in the main window. Instead, at startup it scans the `oswald_globe/tools` package for Python modules and loads any concrete `Tool` subclasses it finds.

It also supports a parallel plugin system for whole-globe filters. These are discovered from `oswald_globe/filters` and appear under the app's `Filters` menu.

If you add a new tool file to `oswald_globe/tools`, the app will discover it automatically when the application starts.

## Where to put a plugin

Create a new Python file in:

`oswald_globe/tools/`

For example:

`oswald_globe/tools/my_tool.py`

The file may contain:

- one `Tool` subclass
- one or more helper classes
- one optional Qt options dialog class
- one optional undoable command or brush command class

The file name can be anything you like, as long as it does not start with `_`.

## What the application discovers

The loader looks for any class that:

- is a subclass of `Tool`
- is not the abstract base `Tool` itself
- is defined in the module being imported
- is not abstract

It normalizes the tool name from either the `_menu_label` or the class name, and then creates a menu action for it.

## Required base classes

The common base classes live in `oswald_globe/tools/tool.py`.

### `Tool`

Every tool plugin should inherit from `Tool`.

Required methods:

- `create_menu_action(self, tools_menu, action_group, on_selected) -> QAction`
  - Creates the menu entry in the Tools menu.
  - Must add the action to `tools_menu` and `action_group`.
  - Usually sets `self._menu_action = action`.

Optional methods:

- `create_options_dialog(self) -> Optional[QDialog]`
  - Returns a Qt dialog for tool settings.
  - Called when the tool is activated and the dialog is needed.

- `show_options_dialog(self)`
  - Built into the base class; normally you do not override it.

- `dispose_options_dialog(self)`
  - Built into the base class; normally you do not override it.

- `apply_interaction(self, gl_widget, pt, *, replace_existing=False, erase_selection=False) -> None`
  - Lets the tool describe what should happen when the user clicks or drags over the globe.
  - Use this for custom logic when your tool does not fit the built-in brush pattern.

### `BrushTool`

Use this for tools that operate like paint/select brushes.

It adds a brush-based interface:

- `_menu_label` = text shown in the Tools menu
- `radius_km()`
- `falloff_percent()`
- `create_brush_command(payload, gl_widget)`
- `apply_interaction(...)`

The default brush interaction flow is:

1. the globe widget computes the target lat/lon point
2. the active tool receives the clicked point
3. the tool creates a payload dictionary
4. `create_brush_command(...)` returns an undoable command
5. the command is pushed onto the undo stack

This is the main extension point for brush-like tools.

## Minimum example: a simple tool

```python
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from oswald_globe.tools.tool import Tool


class MyTool(Tool):
    _menu_label = "My Tool"

    def __init__(self, parent, on_brush_changed=None):
        super().__init__(parent)
        self._on_brush_changed = on_brush_changed

    def create_menu_action(self, tools_menu: QMenu, action_group: QActionGroup, on_selected):
        action = QAction(self._menu_label, self._parent)
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False: on_selected())
        tools_menu.addAction(action)
        action_group.addAction(action)
        self._menu_action = action
        return action
```

This is enough for the app to discover the tool and add it to the Tools menu.

## Example: a brush tool

This is the pattern used by the built-in `PaintTool` and `SelectTool`.

```python
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QDialog, QWidget

from oswald_globe.tools.tool import BrushTool


class MyBrushToolOptionsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("My Tool Options")
        self.resize(300, 120)


class MyBrushTool(BrushTool):
    _menu_label = "My Brush"

    def __init__(self, parent: QWidget, on_brush_changed):
        super().__init__(parent, on_brush_changed)
        self._radius_km = 200
        self._falloff_percent = 35

    def create_options_dialog(self) -> Optional[QDialog]:
        dialog = MyBrushToolOptionsDialog(self._parent)
        return dialog

    def create_brush_command(self, payload: Dict[str, Any], gl_widget: Any):
        # Create your undoable command here.
        # For example, a subclass of BrushCommand or any command with redo()/undo().
        return MyBrushCommand(
            target_lat_deg=float(payload["target_lat_deg"]),
            target_lon_deg=float(payload["target_lon_deg"]),
            radius_km=float(payload["radius_km"]),
            falloff_percent=float(payload["falloff_percent"]),
            mesh_grid=gl_widget.mesh_grid,
            # ... other parameters
        )
```

## Creating a custom options dialog

Tool settings should usually live in the same file as the tool class. This mirrors how the built-in paint and select tools are organized.

A dialog can subclass `QDialog` directly, or subclass the shared `ToolOptionsDialog` in `oswald_globe/tools/tool_options_dialog.py` if you want to reuse the slider/spinbox pattern.

Typical dialog responsibilities:

- expose tool parameters
- update the tool state when the user changes values
- call the callback passed in as `on_brush_changed` when brush radius/falloff changes

For example, the built-in tool dialogs set a radius slider, falloff slider, and optional mode/value controls.

## Brush commands and undo/redo

Tool logic is often implemented as an undoable command, which is pushed onto the app’s `UndoStack`.

A command should support:

- `redo()`
- `undo()`

The built-in paint/select commands live next to their tool code and are created by the tool’s `create_brush_command()` method.

This architectural split keeps the tool’s behavior co-located with the tool itself, which makes plugins more self-contained and easier to write.

## Filter plugins

Filter plugins follow the same discovery pattern as tools, but they live in `oswald_globe/filters/` and are added under the `Filters` menu. A filter is a whole-selection operation: it is applied to the current mesh selection and uses the per-vertex selection value as a blend weight.

If there is no selection, the effective selection is treated as all vertices selected.

The built-in filter pattern is:

- a subclass of `Filter`
- a `create_menu_action(self, filters_menu, action_group, on_selected)` method
- an optional `create_options_dialog()` method
- an `apply(self, gl_widget)` method that pushes an undoable command or executes the edit

A simple example is the built-in `FillFilter`, which reuses the paint value/mode controls and then applies the chosen operation to all selected vertices (or every vertex if there is no selection).

## Custom logic when you do not want a brush command

If your tool does something different from a brush operation, override `apply_interaction()` directly.

That method receives:

- `gl_widget`: the OpenGL globe widget
- `pt`: the hovered/intersected point dictionary, or `None`
- `replace_existing`: optional mode flag
- `erase_selection`: optional mode flag

This is a good place to implement:

- selection tools with custom rules
- region-based tools
- filtering tools
- non-brush interactions

## Important rules for plugin files

- Put the new file in `oswald_globe/tools/`
- Use a concrete subclass of `Tool` or `BrushTool`
- Do not rely on the main app importing the class explicitly
- The application discovers it automatically from the package directory
- Keep the tool dialog and the tool logic in the same file when possible

## Example directory layout

```text
oswald_globe/
  tools/
    __init__.py
    tool.py
    tool_options_dialog.py
    paint_tool.py
    select_tool.py
    my_custom_tool.py
```

The `my_custom_tool.py` file can contain:

```python
from PySide6.QtWidgets import QDialog

from oswald_globe.tools.tool import BrushTool


class MyCustomToolOptionsDialog(QDialog):
    ...


class MyCustomTool(BrushTool):
    _menu_label = "Custom"

    def __init__(self, parent, on_brush_changed):
        super().__init__(parent, on_brush_changed)

    def create_menu_action(self, tools_menu, action_group, on_selected):
        ...

    def create_options_dialog(self):
        return MyCustomToolOptionsDialog(self._parent)

    def create_brush_command(self, payload, gl_widget):
        ...
```

The app will pick it up automatically when it starts.

## Naming convention

The plugin system uses either:

- the `_menu_label` attribute, or
- the class name

If your class is called `MyTool`, the menu label will be `My` or `My Tool` depending on how you set it. In practice, setting `_menu_label` is the clearest choice.

## Summary

To create a new tool:

1. add a new Python module under `oswald_globe/tools`
2. define a subclass of `Tool` or `BrushTool`
3. implement `create_menu_action(...)`
4. optionally implement `create_options_dialog(...)`
5. optionally implement `create_brush_command(...)` or `apply_interaction(...)`
6. drop the file into the tools directory and restart the app

The application will discover it automatically without changes to the main application code.
