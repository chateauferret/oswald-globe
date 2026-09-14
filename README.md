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

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Launch the default sea-level globe:
```bash
python app.py
```

Or load a custom heightfield raster:
```bash
python app.py --heightfield path/to/heightfield.tif --mesh-min-level 2 --mesh-max-level 7 --mesh-threshold 100.0
```

## Running Tests

```bash
pytest tests
```
