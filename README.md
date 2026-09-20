# Oswald Globe

Desktop interactive 3D virtual globe application built with **PySide6 (Qt)** and **PyOpenGL**.

## Features

- **Interactive 3D Globe**: Smooth rotation, panning, inertia, and orthographic zoom.
- **Adaptive Icosphere Mesh**: Geodesic sphere subdivision driven by terrain elevation gradients with dual (Voronoi) cell wireframe rendering.
- **GLSL Shaders**: Direct hardware acceleration with custom lighting, atmospheric rim glow, and adaptive latitude/longitude graticule lines.
- **On-Screen Floating Controls**: Zoom in/out, view reset, auto-spin toggle, graticule toggle, and mesh wireframe toggle.
- **Dynamic Legend Menu**: Switch globe legends at runtime from the View > Legend menu, with the current legend marked and new resource legends picked up when the menu opens.
- **Live Tooltip**: Dynamic unprojection and elevation lookup on hover.
- **QSS Styling**: Clean, dark-mode native desktop UI.

## Installation

The `run.sh` script creates the local `.venv` when needed, installs the
dependencies from `requirements.txt`, activates the environment, and starts
the application:

```bash
./run.sh
```

Dependencies are checked against `requirements.txt` on each start and are
refreshed automatically when that file changes. To set up the environment
without starting the application, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Startup output is also appended to `.logs/launch.log`, which is useful when
the terminal's scrollback does not show the complete traceback.

## Running the Application

Launch the default sea-level globe:
```bash
./run.sh
```

Or load a custom heightfield raster:
```bash
./run.sh --heightfield path/to/heightfield.tif --mesh-min-level 2 --mesh-max-level 7 --mesh-threshold 100.0
```

## Running Tests

```bash
pytest tests
```
