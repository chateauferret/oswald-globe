"""Seeded 3D Simplex noise sampling for unit-sphere geolocations."""

from __future__ import annotations

import math
import operator

import numpy as np


class GeoSimplexNoise:
    """Sample fractal 3D Simplex noise at a latitude/longitude on the unit sphere."""

    _F3 = 1.0 / 3.0
    _G3 = 1.0 / 6.0
    _GRAD3 = np.asarray(
        (
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (-1.0, 0.0, -1.0),
            (0.0, 1.0, 1.0),
            (0.0, -1.0, 1.0),
            (0.0, 1.0, -1.0),
            (0.0, -1.0, -1.0),
        ),
        dtype=np.float64,
    )

    @staticmethod
    def unit_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
        """Convert a geolocation in degrees to unit-sphere xyz coordinates."""
        lat_rad = math.radians(float(lat_deg))
        lon_rad = math.radians(float(lon_deg))
        cos_lat = math.cos(lat_rad)
        return np.asarray(
            (
                cos_lat * math.cos(lon_rad),
                cos_lat * math.sin(lon_rad),
                math.sin(lat_rad),
            ),
            dtype=np.float64,
        )

    @classmethod
    def sample(
        cls,
        lat_deg: float,
        lon_deg: float,
        *,
        minimum_value: float,
        maximum_value: float,
        frequency: float,
        persistence: float,
        lacunarity: float,
        octaves: int,
        seed: int,
        billow: bool = False,
    ) -> float:
        """Return fractal 3D Simplex noise mapped into the requested numeric range."""
        min_value = cls._validate_ranged_float("minimum_value", minimum_value, -32767.0, 32767.0)
        max_value = cls._validate_ranged_float("maximum_value", maximum_value, -32767.0, 32767.0)
        if min_value > max_value:
            raise ValueError(
                f"minimum_value must be less than or equal to maximum_value, got {min_value} > {max_value}."
            )
        base_frequency = cls._validate_ranged_float("frequency", frequency, 0.0, 100.0)
        persistence_value = cls._validate_ranged_float("persistence", persistence, 0.0, 5.0)
        lacunarity_value = cls._validate_ranged_float("lacunarity", lacunarity, 0.0, 5.0)
        octave_count = cls._validate_ranged_int("octaves", octaves, 0, 20)
        seed_value = operator.index(seed)
        billow_value = bool(billow)

        if octave_count == 0:
            return float((min_value + max_value) * 0.5)

        coordinates = cls.unit_xyz(lat_deg, lon_deg)
        permutation = cls._permutation_table(seed_value)

        total = 0.0
        amplitude = 1.0
        amplitude_sum = 0.0
        current_frequency = base_frequency
        for _ in range(octave_count):
            total += amplitude * cls._noise3d(
                coordinates[0] * current_frequency,
                coordinates[1] * current_frequency,
                coordinates[2] * current_frequency,
                permutation,
            )
            amplitude_sum += amplitude
            amplitude *= persistence_value
            current_frequency *= lacunarity_value

        normalized_noise = 0.0 if amplitude_sum <= 0.0 else float(np.clip(total / amplitude_sum, -1.0, 1.0))
        midpoint = 0.5 * (min_value + max_value)
        half_span = 0.5 * (max_value - min_value)
        centered_noise = normalized_noise
        if billow_value:
            centered_noise = 2.0 * abs(normalized_noise) - 1.0
        return float(midpoint + centered_noise * half_span)

    @classmethod
    def sample_many(
        cls,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        *,
        minimum_value: float,
        maximum_value: float,
        frequency: float,
        persistence: float,
        lacunarity: float,
        octaves: int,
        seed: int,
        billow: bool = False,
    ) -> np.ndarray:
        """Return 3D Simplex noise values for arrays of geolocations."""
        latitudes = np.asarray(lat_deg, dtype=np.float64)
        longitudes = np.asarray(lon_deg, dtype=np.float64)
        if latitudes.shape != longitudes.shape:
            raise ValueError(
                "Latitude and longitude arrays must have matching shapes, "
                f"got {latitudes.shape} and {longitudes.shape}."
            )

        min_value = cls._validate_ranged_float("minimum_value", minimum_value, -32767.0, 32767.0)
        max_value = cls._validate_ranged_float("maximum_value", maximum_value, -32767.0, 32767.0)
        if min_value > max_value:
            raise ValueError(
                f"minimum_value must be less than or equal to maximum_value, got {min_value} > {max_value}."
            )
        base_frequency = cls._validate_ranged_float("frequency", frequency, 0.0, 100.0)
        persistence_value = cls._validate_ranged_float("persistence", persistence, 0.0, 5.0)
        lacunarity_value = cls._validate_ranged_float("lacunarity", lacunarity, 0.0, 5.0)
        octave_count = cls._validate_ranged_int("octaves", octaves, 0, 20)
        seed_value = operator.index(seed)
        billow_value = bool(billow)

        if octave_count == 0:
            midpoint = (min_value + max_value) * 0.5
            return np.full(latitudes.shape, midpoint, dtype=np.float64)

        permutation = cls._permutation_table(seed_value)
        values = np.empty(latitudes.shape, dtype=np.float64)
        flat_latitudes = latitudes.ravel()
        flat_longitudes = longitudes.ravel()
        flat_values = values.ravel()
        raw_noise = np.empty(flat_latitudes.shape, dtype=np.float64)

        for index, (latitude, longitude) in enumerate(zip(flat_latitudes, flat_longitudes)):
            coordinates = cls.unit_xyz(float(latitude), float(longitude))
            total = 0.0
            amplitude = 1.0
            amplitude_sum = 0.0
            current_frequency = base_frequency
            for _ in range(octave_count):
                total += amplitude * cls._noise3d(
                    coordinates[0] * current_frequency,
                    coordinates[1] * current_frequency,
                    coordinates[2] * current_frequency,
                    permutation,
                )
                amplitude_sum += amplitude
                amplitude *= persistence_value
                current_frequency *= lacunarity_value

            raw_noise[index] = 0.0 if amplitude_sum <= 0.0 else float(np.clip(total / amplitude_sum, -1.0, 1.0))

        if billow_value:
            raw_noise = np.abs(raw_noise)
            min_noise = float(np.min(raw_noise))
            max_noise = float(np.max(raw_noise))
            if max_noise > min_noise:
                raw_noise = 2.0 * ((raw_noise - min_noise) / (max_noise - min_noise)) - 1.0
            else:
                raw_noise.fill(0.0)
        else:
            max_abs = float(np.max(np.abs(raw_noise)))
            if max_abs > 0.0:
                raw_noise = raw_noise / max_abs

        midpoint = 0.5 * (min_value + max_value)
        half_span = 0.5 * (max_value - min_value)
        values[:] = midpoint + raw_noise.reshape(values.shape) * half_span
        return values

    @staticmethod
    def _validate_ranged_float(name: str, value: float, minimum: float, maximum: float) -> float:
        numeric_value = float(value)
        if not minimum <= numeric_value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}, got {numeric_value}.")
        return numeric_value

    @staticmethod
    def _validate_ranged_int(name: str, value: int, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {value!r}.")
        int_value = operator.index(value)
        if not minimum <= int_value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}, got {int_value}.")
        return int_value

    @staticmethod
    def _permutation_table(seed: int) -> np.ndarray:
        normalized_seed = np.uint64(seed % (1 << 64))
        rng = np.random.default_rng(normalized_seed)
        permutation = rng.permutation(256).astype(np.int16, copy=False)
        return np.concatenate((permutation, permutation))

    @classmethod
    def _noise3d(cls, x: float, y: float, z: float, permutation: np.ndarray) -> float:
        skew = (x + y + z) * cls._F3
        i = math.floor(x + skew)
        j = math.floor(y + skew)
        k = math.floor(z + skew)
        unskew = (i + j + k) * cls._G3
        x0 = x - (i - unskew)
        y0 = y - (j - unskew)
        z0 = z - (k - unskew)

        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1 = 1, 0, 0
                i2, j2, k2 = 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1 = 1, 0, 0
                i2, j2, k2 = 1, 0, 1
            else:
                i1, j1, k1 = 0, 0, 1
                i2, j2, k2 = 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1 = 0, 0, 1
                i2, j2, k2 = 0, 1, 1
            elif x0 < z0:
                i1, j1, k1 = 0, 1, 0
                i2, j2, k2 = 0, 1, 1
            else:
                i1, j1, k1 = 0, 1, 0
                i2, j2, k2 = 1, 1, 0

        x1 = x0 - i1 + cls._G3
        y1 = y0 - j1 + cls._G3
        z1 = z0 - k1 + cls._G3
        x2 = x0 - i2 + 2.0 * cls._G3
        y2 = y0 - j2 + 2.0 * cls._G3
        z2 = z0 - k2 + 2.0 * cls._G3
        x3 = x0 - 1.0 + 3.0 * cls._G3
        y3 = y0 - 1.0 + 3.0 * cls._G3
        z3 = z0 - 1.0 + 3.0 * cls._G3

        ii = i & 255
        jj = j & 255
        kk = k & 255
        gi0 = permutation[ii + permutation[jj + permutation[kk]]] % 12
        gi1 = permutation[ii + i1 + permutation[jj + j1 + permutation[kk + k1]]] % 12
        gi2 = permutation[ii + i2 + permutation[jj + j2 + permutation[kk + k2]]] % 12
        gi3 = permutation[ii + 1 + permutation[jj + 1 + permutation[kk + 1]]] % 12

        n0 = cls._corner_contribution(gi0, x0, y0, z0)
        n1 = cls._corner_contribution(gi1, x1, y1, z1)
        n2 = cls._corner_contribution(gi2, x2, y2, z2)
        n3 = cls._corner_contribution(gi3, x3, y3, z3)
        return 32.0 * (n0 + n1 + n2 + n3)

    @classmethod
    def _corner_contribution(cls, gradient_index: int, x: float, y: float, z: float) -> float:
        attenuation = 0.6 - x * x - y * y - z * z
        if attenuation <= 0.0:
            return 0.0
        gradient = cls._GRAD3[int(gradient_index)]
        attenuation *= attenuation
        return attenuation * attenuation * float(gradient[0] * x + gradient[1] * y + gradient[2] * z)


__all__ = ["GeoSimplexNoise"]
