from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QDockWidget, QMainWindow, QWidget

from oswald_globe.filters.fill_filter import FillFilter, FillFilterOptionsDialog
from oswald_globe.icosphere_grid import IcosphereGrid
from oswald_globe.tools.paint_tool import PaintToolOptionsDialog
from oswald_globe.tools.select_tool import SelectToolOptionsDialog
from oswald_globe.undo_stack import UndoStack


@pytest.mark.parametrize(
    "panel_class", [PaintToolOptionsDialog, SelectToolOptionsDialog, FillFilterOptionsDialog]
)
def test_options_panels_reflow_at_each_edge_and_retain_layout_when_floating(qapp, panel_class):
    window = QMainWindow()
    window.resize(1000, 600)
    panel = panel_class(window)
    try:
        assert isinstance(panel, QDockWidget)
        assert panel.allowedAreas() == Qt.DockWidgetArea.AllDockWidgetAreas
        assert panel.features() & QDockWidget.DockWidgetFeature.DockWidgetMovable
        assert panel.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
        assert window.dockWidgetArea(panel) == Qt.DockWidgetArea.RightDockWidgetArea
        panel.radius_spin.setValue(321)
        panel.falloff_spin.setValue(12)
        panel.value_spin.setValue(73)
        panel.mode_combo.setCurrentText("Multiply")
        window.show()
        panel.show()
        for area, horizontal in (
            (Qt.DockWidgetArea.TopDockWidgetArea, True),
            (Qt.DockWidgetArea.LeftDockWidgetArea, False),
            (Qt.DockWidgetArea.BottomDockWidgetArea, True),
            (Qt.DockWidgetArea.RightDockWidgetArea, False),
        ):
            panel.setFloating(False)
            window.addDockWidget(area, panel)
            qapp.processEvents()
            layout = panel.widget().layout()
            positions = [
                layout.getItemPosition(layout.indexOf(control))
                for _, control, _ in panel._rows
            ]
            if horizontal:
                assert {row for row, _, _, _ in positions} == {1}
                assert [column for _, column, _, _ in positions] == list(range(0, 2 * len(positions), 2))
                assert panel.sizeHint().height() < 100
            else:
                assert [row for row, _, _, _ in positions] == list(range(len(positions)))
                assert {column for _, column, _, _ in positions} == {1}
                assert panel.sizeHint().height() < 200
            if isinstance(panel, FillFilterOptionsDialog):
                assert panel._apply_button.isVisible()
            panel.setFloating(True)
            qapp.processEvents()
            assert panel.isFloating()
            assert positions == [
                layout.getItemPosition(layout.indexOf(control))
                for _, control, _ in panel._rows
            ]
            assert panel.radius_slider.value() == 321
            assert panel.falloff_slider.value() == 12
            assert panel.value_slider.value() == 73
            assert panel.mode_combo.currentText() == "Multiply"
    finally:
        panel.close()
        window.close()
        window.deleteLater()


def test_fill_panel_apply_ok_cancel_and_reopen_preserve_behavior(qapp):
    window = QMainWindow()
    grid = IcosphereGrid()
    grid.set_layer("elevation", np.zeros(grid.vertex_count(), dtype=np.float64))
    undo_stack = UndoStack()
    window.viewer = SimpleNamespace(
        gl_widget=SimpleNamespace(
            mesh_grid=grid,
            _undo_stack=undo_stack,
            apply_mesh_vertex_values=lambda indices, values: grid.get_layer("elevation").values.__setitem__(
                indices, values
            ),
        )
    )
    filter_obj = FillFilter(window)
    try:
        window.show()
        filter_obj.show_options_dialog()
        panel = filter_obj.options_dialog
        buttons = panel.findChild(QDialogButtonBox)
        panel.value_spin.setValue(25)
        panel._apply_button.click()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 25)
        assert panel.isVisible()
        undo_stack.undo()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 0)
        undo_stack.redo()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 25)

        panel.value_spin.setValue(50)
        buttons.button(QDialogButtonBox.StandardButton.Ok).click()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 50)
        assert panel.isHidden()
        filter_obj.show_options_dialog()
        assert filter_obj.options_dialog is panel
        assert panel.value_spin.value() == 50
        panel.value_spin.setValue(99)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 50)
        undo_stack.undo()
        np.testing.assert_allclose(grid.get_layer("elevation").values, 25)

        filter_obj.dispose_options_dialog()
        assert filter_obj.options_dialog is None
        assert window.dockWidgetArea(panel) == Qt.DockWidgetArea.NoDockWidgetArea
        filter_obj.show_options_dialog()
        assert filter_obj.options_dialog.value_spin.value() == 99
    finally:
        filter_obj.dispose_options_dialog()
        window.close()
        window.deleteLater()


@pytest.mark.parametrize(
    "panel_class", [PaintToolOptionsDialog, SelectToolOptionsDialog, FillFilterOptionsDialog]
)
@pytest.mark.parametrize(
    "area",
    [
        Qt.DockWidgetArea.TopDockWidgetArea,
        Qt.DockWidgetArea.BottomDockWidgetArea,
        Qt.DockWidgetArea.LeftDockWidgetArea,
        Qt.DockWidgetArea.RightDockWidgetArea,
    ],
)
def test_panel_controls_stay_fixed_and_floating_window_fits_contents(qapp, panel_class, area):
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.resize(1000, 600)
    panel = panel_class(window)
    window.addDockWidget(area, panel)
    try:
        window.show()
        panel.show()
        qapp.processEvents()
        controls = [
            widget
            for _, control, spin in panel._rows
            for widget in (control, spin)
            if widget is not None
        ]
        sizes = [control.size() for control in controls]
        window.resize(1600, 1000)
        qapp.processEvents()
        assert [control.size() for control in controls] == sizes

        panel.setFloating(True)
        qapp.processEvents()
        content = panel.widget()
        assert content.size() == content.layout().sizeHint()
        assert [control.size() for control in controls] == sizes
        compact_size = panel.size()
        panel.resize(1600, 1000)
        qapp.processEvents()
        assert panel.size() == compact_size

        panel.setFloating(False)
        window.addDockWidget(area, panel)
        qapp.processEvents()
        assert [control.size() for control in controls] == sizes
        if area in (Qt.DockWidgetArea.TopDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea):
            assert panel.width() == window.width()
        else:
            assert panel.height() > compact_size.height()
    finally:
        panel.close()
        window.close()
        window.deleteLater()
