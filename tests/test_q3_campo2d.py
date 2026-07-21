from __future__ import annotations

import csv
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from heliostat.q3_campo2d.evaluate import evaluate_design, smoke_profile
from heliostat.q3_campo2d.model import (
    Campo2DDesign,
    build_campo_field,
    expand_design,
    load_q2_campo_base,
    validate_heterogeneous_field,
)
from heliostat.q3_campo2d.search import (
    _sobol_designs,
    angle_candidates,
    radial_height_candidates,
    radial_size_candidates,
    ring_candidates,
    scalar_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
Q2_SUMMARY = ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
Q2_COORDINATES = ROOT / "outputs" / "q2" / "03_最终镜位坐标.csv"


class Campo2DModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_q2_campo_base(Q2_SUMMARY, Q2_COORDINATES)
        cls.uniform = Campo2DDesign.uniform(
            cls.base.parameters,
            ring_count=cls.base.ring_count,
        )
        cls.field = build_campo_field(cls.base, cls.uniform)

    def test_uniform_reconstructs_q2_coordinates_exactly(self) -> None:
        with Q2_COORDINATES.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = np.asarray(
            [[float(row["x_m"]), float(row["y_m"])] for row in rows]
        )
        self.assertEqual(self.field.mirror_count, 1469)
        np.testing.assert_allclose(self.field.coordinates, expected, atol=0.0)
        self.assertEqual(
            self.base.excluded_ring_angles,
            (
                (28, 0.30543261909900765),
                (28, -0.30543261909900754),
            ),
        )

    def test_control_nodes_follow_new_document(self) -> None:
        self.assertEqual(self.field.control_ring_indices, (1, 7, 12, 20, 28))
        self.assertTrue(np.all(self.field.radial_basis >= -1e-12))
        np.testing.assert_allclose(
            np.sum(self.field.radial_basis, axis=1),
            1.0,
            atol=1e-12,
        )

    def test_angular_features_are_centered_within_each_ring(self) -> None:
        for ring in np.unique(self.field.ring_indices):
            active = self.field.ring_indices == ring
            np.testing.assert_allclose(
                np.mean(self.field.angular_features[active], axis=0),
                (0.0, 0.0),
                atol=1e-14,
            )

    def test_uniform_specifications_reproduce_q2_values(self) -> None:
        specifications = expand_design(self.field, self.uniform)
        np.testing.assert_allclose(
            specifications.widths,
            self.base.parameters.mirror_width,
        )
        np.testing.assert_allclose(
            specifications.heights,
            self.base.parameters.mirror_height,
        )
        np.testing.assert_allclose(
            specifications.installation_heights,
            self.base.parameters.installation_height,
        )
        check = validate_heterogeneous_field(
            field=self.field,
            specifications=specifications,
        )
        self.assertTrue(check.valid, check.reason)

    def test_tower_change_regenerates_campo(self) -> None:
        changed = replace(self.uniform, tower_y=self.uniform.tower_y + 1.0)
        field = build_campo_field(self.base, changed)
        self.assertFalse(np.array_equal(field.coordinates, self.field.coordinates))
        self.assertAlmostEqual(field.parameters.tower_y, changed.tower_y)
        self.assertLessEqual(
            float(np.max(np.hypot(field.coordinates[:, 0], field.coordinates[:, 1]))),
            350.0 + 1e-9,
        )

    def test_search_actions_cover_all_parameter_blocks(self) -> None:
        self.assertEqual(len(radial_height_candidates(self.uniform, 0.1)), 10)
        self.assertEqual(len(radial_size_candidates(self.uniform, 0.01)), 10)
        self.assertEqual(
            len(angle_candidates(self.uniform, target="size", step=0.01)),
            4,
        )
        self.assertEqual(
            len(angle_candidates(self.uniform, target="height", step=0.1)),
            4,
        )
        self.assertEqual(
            len(
                scalar_candidates(
                    self.uniform,
                    parameter="initial_spacing",
                    step=0.1,
                )
            ),
            2,
        )
        self.assertEqual(len(ring_candidates(self.uniform)), 2)

    def test_sobol_initials_are_deterministic(self) -> None:
        first = _sobol_designs(base=self.base, count=4, seed=2023)
        second = _sobol_designs(base=self.base, count=4, seed=2023)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)


class Campo2DEvaluationTests(unittest.TestCase):
    def test_smoke_evaluation_uses_q2_baseline(self) -> None:
        base = load_q2_campo_base(Q2_SUMMARY, Q2_COORDINATES)
        design = Campo2DDesign.uniform(base.parameters, ring_count=base.ring_count)
        evaluation = evaluate_design(
            base=base,
            design=design,
            profile=smoke_profile(),
        )
        self.assertEqual(evaluation.mirror_count, 1469)
        self.assertTrue(evaluation.geometry.valid)
        self.assertGreater(evaluation.annual_power_mw, 0.0)


if __name__ == "__main__":
    unittest.main()
