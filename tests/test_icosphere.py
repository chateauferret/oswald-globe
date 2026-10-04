import numpy as np
import pytest

from oswald_globe.icosphere_grid import IcosphereBuildCancelled, IcosphereGrid, LayerLegend, _xyz_to_latlon


def _bump_map(height=181, width=361):
    """Equirectangular raster with a steep single bump near the equator/prime meridian
    and a flat baseline elsewhere, to exercise adaptive refinement."""
    lat = np.linspace(90, -90, height)
    lon = np.linspace(-180, 180, width)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    dist2 = lat_grid ** 2 + lon_grid ** 2
    return 5000.0 * np.exp(-dist2 / 50.0)


def test_base_icosahedron_shape():
    grid = IcosphereGrid()
    assert grid.vertex_count() == 12
    assert grid.face_count() == 20


def test_uniform_subdivision_counts():
    grid = IcosphereGrid()
    grid.subdivide_uniform(2)
    assert grid.face_count() == 20 * 4 ** 2
    f = grid.face_count()
    expected_v = f // 2 + 2
    assert grid.vertex_count() == expected_v


def test_subdivide_selected_faces_once_refines_selected_triangles():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 1.0

    subdivided_faces = grid.subdivide_selected_faces_once()

    assert subdivided_faces == 20 * 4
    assert grid.face_count() == 20 * 4 ** 2
    assert grid.vertex_count() == 10 * (4 ** 2) + 2
    degrees = np.array([len(neighbors) for neighbors in grid.build_adjacency().values()])
    assert np.count_nonzero(degrees == 5) == 12
    assert np.all((degrees == 5) | (degrees == 6))


def test_repeated_local_subdivision_stays_balanced_and_manifold():
    grid = IcosphereGrid()
    grid.subdivide_uniform(2)
    original_faces = list(grid._red_leaves())
    grid.get_layer(grid.SELECTION_LAYER_NAME).values[list(original_faces[0].v)] = 1.0
    grid.add_layer("constant", np.full(grid.vertex_count(), 42.0))
    grid.dual_cell_line_segments()

    for _ in range(4):
        assert grid.subdivide_selected_faces_once() > 0
        edge_counts: dict[tuple[int, int], int] = {}
        for face in grid.leaf_faces():
            a, b, c = face.v
            for edge in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(edge))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        assert set(edge_counts.values()) == {2}
        assert grid.vertex_count() - len(edge_counts) + grid.face_count() == 2
        for face in grid._red_leaves():
            a, b, c = face.v
            assert all(grid._far_side_depth(i, j) <= 1 for i, j in ((a, b), (b, c), (c, a)))
        _, ranges = grid.dual_cell_line_segments()
        assert np.all((ranges[:, 1] // 2 >= 5) & (ranges[:, 1] // 2 <= 8))
        assert max(map(len, grid.build_adjacency().values())) <= 8
        np.testing.assert_allclose(grid.get_layer("constant").values, 42.0)

    assert any(face.is_leaf for face in original_faces)


def test_local_subdivision_refines_overcrowded_transition_cells():
    grid = IcosphereGrid()
    grid.subdivide_uniform(2)
    selection = grid.get_layer(grid.SELECTION_LAYER_NAME)
    selection.values[:] = 1.0
    selection.values[grid._red_leaves()[0].v[0]] = 0.0

    grid.subdivide_selected_faces_once()

    assert max(map(len, grid.build_adjacency().values())) <= 8
    _, ranges = grid.dual_cell_line_segments()
    assert np.all(ranges[:, 1] // 2 <= 8)


def test_local_subdivision_uses_shared_spherical_edge_midpoints():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    face = grid._red_leaves()[0]
    grid.get_layer(grid.SELECTION_LAYER_NAME).values[list(face.v)] = (0.25, 0.5, 1.0)
    grid.add_layer("temperature", np.arange(grid.vertex_count(), dtype=np.float64))
    original_values = {name: grid.get_layer(name).values.copy() for name in grid.layer_names()}

    grid.subdivide_selected_faces_once()

    a, b, c = grid._face_child_midpoints(face)
    v0, v1, v2 = face.v
    assert [child.v for child in face.children] == [
        (v0, a, c), (a, v1, b), (c, b, v2), (a, b, c)
    ]
    for midpoint, (i, j) in zip((a, b, c), ((v0, v1), (v1, v2), (v2, v0))):
        expected = grid.vertices[i] + grid.vertices[j]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(grid.vertices[midpoint], expected)
        assert grid._edge_midpoints[tuple(sorted((i, j)))] == midpoint
        for name in ("temperature", grid.SELECTION_LAYER_NAME):
            assert grid.get_layer(name).values[midpoint] == (original_values[name][i] + original_values[name][j]) / 2.0
    for name, values in original_values.items():
        np.testing.assert_array_equal(grid.get_layer(name).values[:len(values)], values)


def test_local_subdivision_preserves_existing_midpoint_layer_values():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    face = grid._red_leaves()[0]
    grid.get_layer(grid.SELECTION_LAYER_NAME).values[list(face.v)] = 1.0
    grid.subdivide_selected_faces_once()
    midpoint = grid._face_child_midpoints(face)[0]
    grid.get_layer("elevation").values[midpoint] = 123.0
    grid.get_layer(grid.SELECTION_LAYER_NAME).values[:] = 1.0

    grid.subdivide_selected_faces_once()

    assert grid.get_layer("elevation").values[midpoint] == 123.0


def test_local_subdivision_without_selection_does_not_change_mesh():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    vertices = grid.vertices.copy()
    faces = [face.v for face in grid.leaf_faces()]

    assert grid.subdivide_selected_faces_once() == 0

    np.testing.assert_array_equal(grid.vertices, vertices)
    assert [face.v for face in grid.leaf_faces()] == faces


def test_remove_selected_lowest_level_vertices_coarsens_selected_detail():
    grid = IcosphereGrid()
    grid.subdivide_uniform(2)
    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[:] = 1.0

    removed_vertices = grid.remove_selected_lowest_level_vertices()

    assert removed_vertices > 0
    assert grid.face_count() == 20 * 4
    assert grid.vertex_count() == 10 * (4 ** 1) + 2


def test_adaptive_refinement_denser_near_bump():
    arr = _bump_map()
    grid = IcosphereGrid.from_equirectangular(arr, min_level=1, max_level=6, threshold=200.0)

    lat, lon = grid.vertex_lat_lon()
    near_bump = np.sqrt(lat ** 2 + lon ** 2) < 10.0
    far_from_bump = np.sqrt((lat - 90) ** 2 + lon ** 2) < 10.0

    assert near_bump.sum() > far_from_bump.sum()


def test_round_trip_equirectangular():
    arr = _bump_map(height=91, width=181)
    grid = IcosphereGrid.from_equirectangular(arr, min_level=3, max_level=7, threshold=100.0)

    out = grid.to_equirectangular(height=91, width=181)
    assert out.shape == arr.shape
    assert np.all(np.isfinite(out))

    assert np.corrcoef(arr.ravel(), out.ravel())[0, 1] > 0.9
    assert np.max(np.abs(out - arr)) < 2000.0


def test_build_adjacency_symmetric():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    adjacency = grid.build_adjacency()
    for v, neighbors in adjacency.items():
        for n in neighbors:
            assert v in adjacency[n]


def test_sample_matches_vertex_values_at_vertices():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)
    lat, lon = grid.vertex_lat_lon()
    for i in range(grid.vertex_count()):
        grid._values[i] = float(i)

    sampled = grid.sample(lat, lon)
    assert np.allclose(sampled, np.arange(grid.vertex_count()), atol=1e-6)


def test_layers_support_metadata_and_ordered_storage():
    grid = IcosphereGrid()
    elevation = grid.get_layer("elevation")
    elevation.description = "Terrain height"
    elevation.opacity = 65.0
    elevation.legend = LayerLegend(name="topo", colormap=object())
    elevation.visible = True

    thermal_cmap = object()
    temp = grid.add_layer(
        "temperature",
        np.full(grid.vertex_count(), 12.0, dtype=np.float64),
        description="Surface temperature",
        opacity=80.0,
        legend=LayerLegend(name="thermal", colormap=thermal_cmap),
        z_index=5,
    )

    assert grid.layer_names() == ["elevation", "selection_status", "temperature"]
    assert elevation.name == "elevation"
    assert elevation.description == "Terrain height"
    assert elevation.opacity == 65.0
    assert elevation.legend is not None
    assert elevation.legend.name == "topo"
    assert temp.name == "temperature"
    assert temp.description == "Surface temperature"
    assert temp.opacity == 80.0
    assert temp.legend is not None
    assert temp.legend.name == "thermal"
    assert temp.legend.colormap is thermal_cmap
    assert temp.z_index == 5


def test_selection_layer_defaults_to_zero():
    grid = IcosphereGrid()

    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)

    np.testing.assert_allclose(selection.values, 0.0)
    assert grid.has_selection is False


def test_has_selection_tracks_positive_selection_values():
    grid = IcosphereGrid()

    assert grid.has_selection is False

    selection = grid.get_layer(IcosphereGrid.SELECTION_LAYER_NAME)
    selection.values[0] = 0.25

    assert grid.has_selection is True

    selection.values[:] = 0.0

    assert grid.has_selection is False


def test_dual_cell_line_segments_cover_vertex_cells():
    grid = IcosphereGrid()
    grid.subdivide_uniform(1)

    segments, ranges = grid.dual_cell_line_segments()

    assert segments.ndim == 2
    assert segments.shape[1] == 3
    assert ranges.shape == (grid.vertex_count(), 2)
    assert np.all(ranges[:, 1] > 0)
    assert np.all(ranges[:, 1] % 2 == 0)
    assert np.max(ranges[:, 0] + ranges[:, 1], initial=0) == len(segments)


def test_adaptive_refinement_is_2_1_balanced():
    arr = _bump_map()
    grid = IcosphereGrid.from_equirectangular(arr, min_level=1, max_level=6, threshold=200.0)

    for f in grid._red_leaves():
        v0, v1, v2 = f.v
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            assert grid._far_side_depth(a, b) <= 1


def test_leaf_faces_are_gap_free_manifold():
    arr = _bump_map()
    grid = IcosphereGrid.from_equirectangular(arr, min_level=1, max_level=6, threshold=200.0)

    edge_count: dict[tuple[int, int], int] = {}
    for f in grid.leaf_faces():
        v0, v1, v2 = f.v
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    assert all(count == 2 for count in edge_count.values())


def test_dual_graph_matches_faces_and_is_on_unit_sphere():
    grid = IcosphereGrid()
    grid.subdivide_uniform(2)

    dual_verts = grid.dual_vertices()
    dual_edges = grid.dual_edges()

    assert dual_verts.shape == (grid.face_count(), 3)
    assert np.allclose(np.linalg.norm(dual_verts, axis=1), 1.0, atol=1e-6)
    assert dual_edges.shape[0] == 3 * grid.face_count() // 2
    assert dual_edges.min() >= 0
    assert dual_edges.max() < grid.face_count()


def test_from_equirectangular_reports_creation_and_population_progress():
    arr = _bump_map(height=91, width=181)
    progress_updates: list[tuple[str, int, int]] = []

    grid = IcosphereGrid.from_equirectangular(
        arr,
        min_level=1,
        max_level=4,
        threshold=200.0,
        progress_callback=lambda phase, done, total: progress_updates.append((phase, done, total)),
    )

    creation_updates = [update for update in progress_updates if update[0] == "creating-faces"]
    balance_updates = [update for update in progress_updates if update[0] == "balancing-faces"]
    population_updates = [update for update in progress_updates if update[0] == "populating-faces"]

    assert creation_updates[0] == ("creating-faces", 0, 20 * 4 ** 4)
    assert creation_updates[-1] == ("creating-faces", 20 * 4 ** 4, 20 * 4 ** 4)
    assert balance_updates[0] == ("balancing-faces", 0, 1)
    assert balance_updates[-1][1] == balance_updates[-1][2]
    assert balance_updates[-1][1] >= 1
    assert population_updates[0][0] == "populating-faces"
    assert population_updates[0][1] == 0
    assert population_updates[0][2] == 0
    assert population_updates[1][2] == grid.face_count()
    assert population_updates[-1] == ("populating-faces", grid.face_count(), grid.face_count())


def test_from_equirectangular_can_be_cancelled():
    arr = _bump_map(height=91, width=181)
    cancelled = {"value": False}

    def on_progress(phase: str, done: int, total: int) -> None:
        if phase == "creating-faces" and done > 0:
            cancelled["value"] = True

    with pytest.raises(IcosphereBuildCancelled):
        IcosphereGrid.from_equirectangular(
            arr,
            min_level=1,
            max_level=6,
            threshold=200.0,
            progress_callback=on_progress,
            is_cancelled=lambda: cancelled["value"],
        )
