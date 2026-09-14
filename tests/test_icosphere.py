import numpy as np
import pytest

from oswald_globe.icosphere import IcosphereBuildCancelled, IcosphereGrid, _xyz_to_latlon


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
