import os
import sys
from pathlib import Path
import pytest

# Ensure offscreen Qt platform for testing in headless environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Session-wide QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app
