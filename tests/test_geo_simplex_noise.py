import numpy as np
import pytest

from oswald_globe.algorithms.geo_simplex_noise import GeoSimplexNoise


@pytest.mark.parametrize(
    ("lat_deg", "lon_deg", "expected"),
    [
        (0.0, 0.0, np.array([1.0, 0.0, 0.0])),
        (0.0, 90.0, np.array([0.0, 1.0, 0.0])),
        (90.0, 0.0, np.array([0.0, 0.0, 1.0])),
        (-90.0, 0.0, np.array([0.0, 0.0, -1.0])),
    ],
)
def test_unit_xyz_converts_geolocation_to_unit_sphere(lat_deg: float, lon_deg: float, expected: np.ndarray):
    actual = GeoSimplexNoise.unit_xyz(lat_deg, lon_deg)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.linalg.norm(actual) == pytest.approx(1.0, abs=1e-12)


def test_sample_is_deterministic_for_same_inputs():
    kwargs = dict(
        minimum_value=-5000.0,
        maximum_value=9000.0,
        frequency=3.25,
        persistence=0.6,
        lacunarity=2.0,
        octaves=5,
        seed=12345,
    )

    first = GeoSimplexNoise.sample(12.5, -44.0, **kwargs)
    second = GeoSimplexNoise.sample(12.5, -44.0, **kwargs)

    assert first == pytest.approx(second)


def test_sample_changes_with_seed():
    kwargs = dict(
        minimum_value=-1000.0,
        maximum_value=1000.0,
        frequency=8.0,
        persistence=0.75,
        lacunarity=1.8,
        octaves=4,
    )

    first = GeoSimplexNoise.sample(18.0, 33.0, seed=1, **kwargs)
    second = GeoSimplexNoise.sample(18.0, 33.0, seed=2, **kwargs)

    assert first != pytest.approx(second)


def test_sample_stays_within_requested_range():
    value = GeoSimplexNoise.sample(
        -21.5,
        117.0,
        minimum_value=-32767.0,
        maximum_value=32767.0,
        frequency=12.0,
        persistence=0.8,
        lacunarity=2.5,
        octaves=6,
        seed=-987654321,
    )

    assert -32767.0 <= value <= 32767.0


def test_zero_octaves_returns_range_midpoint():
    value = GeoSimplexNoise.sample(
        51.0,
        -0.1,
        minimum_value=-100.0,
        maximum_value=300.0,
        frequency=10.0,
        persistence=1.0,
        lacunarity=2.0,
        octaves=0,
        seed=42,
    )

    assert value == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"minimum_value": -40000.0, "maximum_value": 1.0, "frequency": 1.0, "persistence": 1.0, "lacunarity": 1.0, "octaves": 1, "seed": 0}, ValueError, "minimum_value"),
        ({"minimum_value": 10.0, "maximum_value": 5.0, "frequency": 1.0, "persistence": 1.0, "lacunarity": 1.0, "octaves": 1, "seed": 0}, ValueError, "minimum_value must be less"),
        ({"minimum_value": 0.0, "maximum_value": 1.0, "frequency": 101.0, "persistence": 1.0, "lacunarity": 1.0, "octaves": 1, "seed": 0}, ValueError, "frequency"),
        ({"minimum_value": 0.0, "maximum_value": 1.0, "frequency": 1.0, "persistence": 6.0, "lacunarity": 1.0, "octaves": 1, "seed": 0}, ValueError, "persistence"),
        ({"minimum_value": 0.0, "maximum_value": 1.0, "frequency": 1.0, "persistence": 1.0, "lacunarity": 6.0, "octaves": 1, "seed": 0}, ValueError, "lacunarity"),
        ({"minimum_value": 0.0, "maximum_value": 1.0, "frequency": 1.0, "persistence": 1.0, "lacunarity": 1.0, "octaves": 21, "seed": 0}, ValueError, "octaves"),
    ],
)
def test_sample_rejects_invalid_ranges(kwargs: dict[str, float | int], error_type: type[Exception], message: str):
    with pytest.raises(error_type, match=message):
        GeoSimplexNoise.sample(0.0, 0.0, **kwargs)
