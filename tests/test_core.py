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
from heliostat.q1.solve import evaluate_time


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

    def test_equal_per_mirror_arrays_match_uniform_field(self) -> None:
        coordinates = np.array(
            [[120.0, 0.0], [140.0, 20.0]],
            dtype=float,
        )
        config = FieldConfig()
        uniform = prepare_field(coordinates, config)
        heterogeneous = prepare_field(
            coordinates,
            config,
            mirror_widths=np.full(2, config.mirror_width),
            mirror_heights=np.full(2, config.mirror_height),
            mirror_center_zs=np.full(2, config.mirror_center_z),
        )

        np.testing.assert_allclose(heterogeneous.centers, uniform.centers)
        np.testing.assert_allclose(
            heterogeneous.mirror_areas,
            uniform.mirror_areas,
        )
        self.assertAlmostEqual(
            heterogeneous.total_mirror_area,
            uniform.total_mirror_area,
        )

        solver = SolverConfig(
            shadow_grid_size=3,
            truncation_rays=8,
            calculate_shadow=False,
            calculate_truncation=False,
        )
        uniform_result = evaluate_time(uniform, 6, 12.0, solver)
        heterogeneous_result = evaluate_time(
            heterogeneous,
            6,
            12.0,
            solver,
        )
        self.assertAlmostEqual(
            heterogeneous_result.field_output_mw,
            uniform_result.field_output_mw,
            places=14,
        )
        self.assertAlmostEqual(
            heterogeneous_result.unit_area_output_kw_m2,
            uniform_result.unit_area_output_kw_m2,
            places=14,
        )

    def test_heterogeneous_power_uses_individual_areas(self) -> None:
        coordinates = np.array(
            [[120.0, 0.0], [140.0, 20.0]],
            dtype=float,
        )
        field = prepare_field(
            coordinates,
            FieldConfig(),
            mirror_widths=np.array([4.0, 8.0]),
            mirror_heights=np.array([3.0, 5.0]),
            mirror_center_zs=np.array([3.0, 5.0]),
        )
        solver = SolverConfig(
            shadow_grid_size=3,
            truncation_rays=8,
            calculate_shadow=False,
            calculate_truncation=False,
        )
        result = evaluate_time(field, 6, 12.0, solver)
        solar = calculate_solar_state(6, 12.0, 39.4, 3.0)
        orientation = calculate_orientation(field, solar.direction)
        optical = (
            orientation.cosine_efficiency
            * field.atmospheric_efficiency
            * field.config.reflectivity
        )
        expected_power_kw = float(
            np.sum(solar.dni_kw_m2 * field.mirror_areas * optical)
        )
        expected_optical = float(
            np.average(optical, weights=field.mirror_areas)
        )

        self.assertAlmostEqual(
            result.field_output_mw,
            expected_power_kw / 1000.0,
        )
        self.assertAlmostEqual(
            result.average_optical_efficiency,
            expected_optical,
        )
        self.assertAlmostEqual(field.total_mirror_area, 52.0)


if __name__ == "__main__":
    unittest.main()
