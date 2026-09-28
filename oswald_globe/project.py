"""Project persistence for raster data plus adaptive icosphere state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .icosphere_grid import IcosphereGrid


@dataclass
class Project:
    """Single-file snapshot of a globe raster, mesh geometry, and mesh layers."""

    FILE_SUFFIX = ".ogp"
    SCHEMA_VERSION = 1

    raster: np.ndarray
    grid: IcosphereGrid
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raster = np.asarray(self.raster, dtype=np.float32)
        if raster.ndim != 2:
            raise ValueError(f"Project raster must be 2D, got shape {raster.shape}.")
        if not isinstance(self.grid, IcosphereGrid):
            raise TypeError(f"Project grid must be an IcosphereGrid, got {type(self.grid)!r}.")
        self.raster = raster
        self.metadata = dict(self.metadata)

    @classmethod
    def is_project_path(cls, path: Path) -> bool:
        return Path(path).suffix.lower() == cls.FILE_SUFFIX

    def save(self, path: Path) -> None:
        archive_payload: Dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
            "raster": np.asarray(self.raster, dtype=np.float32),
            "metadata_json": np.asarray(
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
            ),
        }

        serialized_grid = self.grid.to_serialized_state()
        layer_metadata = serialized_grid.pop("layer_metadata")
        archive_payload["grid_layer_metadata_json"] = np.asarray(
            json.dumps(layer_metadata, sort_keys=True, separators=(",", ":"))
        )
        for key, value in serialized_grid.items():
            archive_payload[f"grid_{key}"] = np.asarray(value)

        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **archive_payload)

    @classmethod
    def load(cls, path: Path) -> "Project":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            schema_version = int(np.asarray(archive["schema_version"], dtype=np.int64).item())
            if schema_version != cls.SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported project schema version {schema_version}; "
                    f"expected {cls.SCHEMA_VERSION}."
                )

            raster = np.asarray(archive["raster"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            layer_metadata = json.loads(
                str(np.asarray(archive["grid_layer_metadata_json"]).item())
            )
            grid_state = {
                "vertices": np.asarray(archive["grid_vertices"], dtype=np.float64),
                "face_vertices": np.asarray(archive["grid_face_vertices"], dtype=np.int64),
                "face_levels": np.asarray(archive["grid_face_levels"], dtype=np.int64),
                "face_children": np.asarray(archive["grid_face_children"], dtype=np.int64),
                "root_indices": np.asarray(archive["grid_root_indices"], dtype=np.int64),
                "edge_midpoint_keys": np.asarray(
                    archive["grid_edge_midpoint_keys"], dtype=np.int64
                ),
                "edge_midpoint_values": np.asarray(
                    archive["grid_edge_midpoint_values"], dtype=np.int64
                ),
                "layer_metadata": layer_metadata,
            }
            for descriptor in layer_metadata:
                values_key = str(descriptor["values_key"])
                grid_state[values_key] = np.asarray(
                    archive[f"grid_{values_key}"], dtype=np.float64
                )

        return cls(raster=raster, grid=IcosphereGrid.from_serialized_state(grid_state), metadata=metadata)
