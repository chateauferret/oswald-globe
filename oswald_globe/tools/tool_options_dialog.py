"""Paint tool options dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QLabel,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


class ToolOptionsDialog(QDialog):
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
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(False)
        self.resize(360, 220)

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

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

        row = 0
        if include_value:
            layout.addWidget(QLabel("Value"), row, 0)
            layout.addWidget(self.value_slider, row, 1)
            layout.addWidget(self.value_spin, row, 2)
            row += 1
        else:
            self.value_slider.hide()
            self.value_spin.hide()

        if include_mode:
            layout.addWidget(QLabel("Mode"), row, 0)
            layout.addWidget(self.mode_combo, row, 1, 1, 2)
            row += 1
        else:
            self.mode_combo.hide()

        if include_radius:
            layout.addWidget(QLabel("Radius (km)"), row, 0)
            layout.addWidget(self.radius_slider, row, 1)
            layout.addWidget(self.radius_spin, row, 2)
            row += 1
        else:
            self.radius_slider.hide()
            self.radius_spin.hide()

        if include_falloff:
            layout.addWidget(QLabel("Falloff (%)"), row, 0)
            layout.addWidget(self.falloff_slider, row, 1)
            layout.addWidget(self.falloff_spin, row, 2)
        else:
            self.falloff_slider.hide()
            self.falloff_spin.hide()

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        self.value_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.value_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
