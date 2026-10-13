import numpy as np
import pytest

from oswald_globe.algorithms.apply_value import ApplyValue


@pytest.mark.parametrize(
    ("mode", "value", "expected"),
    [
        ("Replace", 8.0, np.array([8.0, 8.0, 8.0])),
        ("Add", 2.0, np.array([3.0, 6.0, 9.0])),
        ("Subtract", 2.0, np.array([-1.0, 2.0, 5.0])),
        ("Minimum", 4.0, np.array([1.0, 4.0, 4.0])),
        ("Maximum", 4.0, np.array([4.0, 4.0, 7.0])),
        ("Multiply", 2.0, np.array([2.0, 8.0, 14.0])),
        ("Average", 5.0, np.array([3.0, 4.5, 6.0])),
    ],
)
def test_apply_value_transforms_supported_modes(mode: str, value: float, expected: np.ndarray):
    existing = np.array([1.0, 4.0, 7.0], dtype=np.float64)

    transformed = ApplyValue.transform(existing, value, mode)

    np.testing.assert_allclose(transformed, expected)


def test_apply_value_blends_transformed_values_by_weight():
    existing = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    weights = np.array([0.0, 0.25, 1.0], dtype=np.float64)

    applied = ApplyValue.apply(existing, 50.0, "Replace", weights=weights)

    np.testing.assert_allclose(applied, np.array([10.0, 27.5, 50.0], dtype=np.float64))


def test_apply_value_rejects_invalid_weight_shape():
    with pytest.raises(ValueError, match="Apply weights must match"):
        ApplyValue.apply(np.array([1.0, 2.0]), 3.0, "Add", weights=np.array([[1.0, 1.0]]))
