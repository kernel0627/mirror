from __future__ import annotations

import unittest

import numpy as np

from heliostat.config import FieldConfig, SolverConfig
from heliostat.geometry import (
    calculate_orientation,
    maximum_reflection_error,
    prepare_field,
)
from heliostat.shadow import calculate_shadow_blocking_efficiency
from heliostat.solar import calculate_solar_state, day_from_spring_equinox
from heliostat.truncation import calculate_truncation_efficiency


class SolarTests(unittest.TestCase):
    def test_spring_equinox_is_day_zero(self) -> None:
        self.assertEqual(day_from_spring_equinox(3, 21), 0)

    def test_morning_and_afternoon_are_symmetric(self) -> None:
        morning = calculate_solar_state(6, 9.0, 39.4, 3.0)
        afternoon = calculate_solar_state(6, 15.0, 39.4, 3.0)
        self.assertAlmostEqual(morning.direction[0], -afternoon.direction[0])
        self.assertAlmostEqual(morning.direction[1], afternoon.direction[1])
        self.assertAlmostEqual(morning.direction[2], afternoon.direction[2])
        self.assertAlmostEqual(morning.dni_kw_m2, afternoon.dni_kw_m2)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.field = prepare_field(
            np.array([[120.0, 0.0], [140.0, 20.0]], dtype=float),
            FieldConfig(),
        )
        self.sun = calculate_solar_state(6, 12.0, 39.4, 3.0)
        self.orientation = calculate_orientation(self.field, self.sun.direction)

    def test_center_ray_points_to_receiver(self) -> None:
        error = maximum_reflection_error(
            self.field,
            self.orientation,
            self.sun.direction,
        )
        self.assertLess(error, 1e-12)

    def test_single_mirror_has_no_shadow_or_blocking(self) -> None:
        single = prepare_field(np.array([[120.0, 0.0]]), FieldConfig())
        orientation = calculate_orientation(single, self.sun.direction)
        efficiency = calculate_shadow_blocking_efficiency(
            single,
            orientation,
            self.sun.direction,
            SolverConfig(shadow_grid_size=3, truncation_rays=8),
        )
        np.testing.assert_allclose(efficiency, 1.0)

    def test_receiver_radius_does_not_reduce_truncation(self) -> None:
        small_field = prepare_field(
            np.array([[120.0, 0.0]]),
            FieldConfig(receiver_radius=3.0),
        )
        large_field = prepare_field(
            np.array([[120.0, 0.0]]),
            FieldConfig(receiver_radius=5.0),
        )
        solver = SolverConfig(shadow_grid_size=3, truncation_rays=256)
        small_orientation = calculate_orientation(
            small_field,
            self.sun.direction,
        )
        large_orientation = calculate_orientation(
            large_field,
            self.sun.direction,
        )
        small = calculate_truncation_efficiency(
            small_field,
            small_orientation,
            self.sun.direction,
            solver,
        )
        large = calculate_truncation_efficiency(
            large_field,
            large_orientation,
            self.sun.direction,
            solver,
        )
        self.assertGreaterEqual(float(large[0]), float(small[0]))


if __name__ == "__main__":
    unittest.main()
