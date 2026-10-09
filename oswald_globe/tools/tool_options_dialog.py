"""Compact, dockable tool and filter options."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDockWidget,
    QGridLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


class ToolOptionsDialog(QDockWidget):
    """Options panel; the legacy class name is retained for plugins."""

    _PAINT_MODES = ("Replace", "Add", "Subtract", "Minimum", "Maximum", "Multiply", "Average")

    class _SliderBoundSpinBox(QSpinBox):
        def __init__(
            self,
            slider: QSlider,
            parent: Optional[QWidget] = None,
            *,
            single_step: Optional[int] = None,
        ):
            super().__init__(parent)
            self._slider = slider
            self.setRange(slider.minimum(), slider.maximum())
            if single_step is None:
                step = max(1, int(round((slider.maximum() - slider.minimum()) * 0.05)))
            else:
                step = max(1, int(single_step))
            self.setSingleStep(step)
            self.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            self.setKeyboardTracking(False)
            self.setValue(slider.value())

            slider.valueChanged.connect(self.setValue)
            self.valueChanged.connect(slider.setValue)

        def interpretText(self) -> None:
            editor = self.lineEdit()
            if editor is None:
                return
            text = editor.text().strip()
            if text in {"", "+", "-"}:
                editor.setText(self.textFromValue(self.value()))
                return
            try:
                value = int(text)
            except ValueError:
                editor.setText(self.textFromValue(self.value()))
                return
            if self.minimum() <= value <= self.maximum():
                self.setValue(value)
            else:
                editor.setText(self.textFromValue(self.value()))

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        title: str = "Paint Tool Options",
        value: int = 0,
        mode: str = "Replace",
        radius_km: int = 100,
        falloff_percent: int = 50,
        include_value: bool = True,
        include_mode: bool = True,
        include_radius: bool = True,
        include_falloff: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName(title.replace(" ", ""))
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._horizontal = False
        self._content = QWidget(self)
        self._controls_layout = QGridLayout(self._content)
        self._controls_layout.setContentsMargins(6, 6, 6, 6)
        self._controls_layout.setSpacing(4)
        self._controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._content)
        self._rows: list[tuple[QLabel, QWidget, Optional[QWidget]]] = []
        self._action_widget: Optional[QWidget] = None

        self.value_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.value_slider.setRange(-32767, 32767)
        self.value_slider.setValue(int(value))
        self.value_spin = self._SliderBoundSpinBox(self.value_slider, self, single_step=256)

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(list(self._PAINT_MODES))
        self.mode_combo.setCurrentText(mode if mode in self._PAINT_MODES else self._PAINT_MODES[0])

        self.radius_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.radius_slider.setRange(0, 1000)
        self.radius_slider.setValue(int(radius_km))
        self.radius_spin = self._SliderBoundSpinBox(self.radius_slider, self)

        self.falloff_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.falloff_slider.setRange(0, 100)
        self.falloff_slider.setValue(int(falloff_percent))
        self.falloff_spin = self._SliderBoundSpinBox(self.falloff_slider, self)

        if include_value:
            self._rows.append((QLabel("Value"), self.value_slider, self.value_spin))
        else:
            self.value_slider.hide()
            self.value_spin.hide()

        if include_mode:
            self._rows.append((QLabel("Mode"), self.mode_combo, None))
        else:
            self.mode_combo.hide()

        if include_radius:
            self._rows.append((QLabel("Radius (km)"), self.radius_slider, self.radius_spin))
        else:
            self.radius_slider.hide()
            self.radius_spin.hide()

        if include_falloff:
            self._rows.append((QLabel("Falloff (%)"), self.falloff_slider, self.falloff_spin))
        else:
            self.falloff_slider.hide()
            self.falloff_spin.hide()

        for slider in (self.value_slider, self.radius_slider, self.falloff_slider):
            slider.setFixedWidth(120)
            slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for control in (self.value_spin, self.radius_spin, self.falloff_spin, self.mode_combo):
            control.setFixedSize(control.sizeHint())
        self._relayout_controls()
        self.dockLocationChanged.connect(self._dock_location_changed)
        self.topLevelChanged.connect(self._sync_content_size)
        if isinstance(parent, QMainWindow):
            parent.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self)

    def set_action_widget(self, widget: QWidget) -> None:
        widget.setFixedSize(widget.sizeHint())
        self._action_widget = widget
        self._relayout_controls()

    def _dock_location_changed(self, area: Qt.DockWidgetArea) -> None:
        # Floating reports NoDockWidgetArea; retain the last docked arrangement.
        if area == Qt.DockWidgetArea.NoDockWidgetArea:
            return
        self._horizontal = area in (
            Qt.DockWidgetArea.TopDockWidgetArea,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self._relayout_controls()

    def _relayout_controls(self) -> None:
        layout = self._controls_layout
        while layout.count():
            layout.takeAt(0)
        for column in range(layout.columnCount()):
            layout.setColumnStretch(column, 0)
        for index, (label, control, spin) in enumerate(self._rows):
            if self._horizontal:
                column = 2 * index
                layout.addWidget(label, 0, column, 1, 2)
                layout.addWidget(control, 1, column, 1, 1 if spin is not None else 2)
                if spin is not None:
                    layout.addWidget(spin, 1, column + 1)
            else:
                layout.addWidget(label, index, 0)
                layout.addWidget(control, index, 1, 1, 1 if spin is not None else 2)
                if spin is not None:
                    layout.addWidget(spin, index, 2)
        if self._action_widget is not None:
            if self._horizontal:
                layout.addWidget(self._action_widget, 0, 2 * len(self._rows), 2, 1)
            else:
                layout.addWidget(self._action_widget, len(self._rows), 0, 1, 3)
        self._sync_content_size()

    def _sync_content_size(self) -> None:
        if self.isFloating():
            self._content.setFixedSize(self._controls_layout.sizeHint())
            self.adjustSize()
        else:
            self._content.setMinimumSize(0, 0)
            self._content.setMaximumSize(16777215, 16777215)
