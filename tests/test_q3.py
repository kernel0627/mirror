from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from heliostat.q3.evaluate import (
    EvaluationCache,
    evaluate_design,
    prepare_candidate,
    smoke_profile,
)
from heliostat.q3.model import load_baseline
from heliostat.q3.sensitivity import specification_perturbations
from heliostat.q3.search import coordinate_search
from heliostat.q3.tower_modes import build_refine_field


class SixGroupRefineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = load_baseline(
            q2_summary_path="outputs/q2/07_最终方案摘要.json",
            six_group_summary_path="src/heliostat/q3/six_group_baseline.json",
        )

    def test_zero_increment_reproduces_mother_field_and_specs(self) -> None:
        field, specifications, check = prepare_candidate(
            baseline=self.baseline,
            design=self.baseline.design,
        )
        self.assertTrue(check.valid, check.reason)
        self.assertEqual(field.mirror_count, 1471)
        self.assertEqual(field.group_counts, (72, 269, 283, 224, 357, 266))
        np.testing.assert_allclose(
            field.coordinates,
            self.baseline.mother.coordinates,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            specifications.widths,
            np.asarray(self.baseline.design.widths)[field.group_indices],
        )
        np.testing.assert_array_equal(
            specifications.heights,
            np.asarray(self.baseline.design.mirror_heights)[field.group_indices],
        )

    def test_mode_b_moves_only_tower(self) -> None:
        design = replace(
            self.baseline.design,
            tower_mode="B",
            tower_y=self.baseline.design.tower_y - 0.5,
        )
        field = build_refine_field(self.baseline, design)
        baseline_field = build_refine_field(self.baseline, self.baseline.design)
        np.testing.assert_array_equal(field.coordinates, baseline_field.coordinates)
        self.assertEqual(field.mirror_set_hash, baseline_field.mirror_set_hash)
        self.assertEqual(field.geometry_center_y, self.baseline.design.tower_y)

    def test_mode_a_rebuilds_around_candidate_tower(self) -> None:
        design = replace(
            self.baseline.design,
            tower_mode="A",
            tower_y=self.baseline.design.tower_y - 0.5,
        )
        field = build_refine_field(self.baseline, design)
        self.assertEqual(field.geometry_center_y, design.tower_y)
        first_ring = field.ring_indices == 1
        baseline_first = self.baseline.mother.ring_indices == 1
        np.testing.assert_allclose(
            field.coordinates[first_ring, 1],
            self.baseline.mother.coordinates[baseline_first, 1] - 0.5,
            rtol=0.0,
            atol=1e-12,
        )

    def test_sensitivity_has_36_single_parameter_candidates(self) -> None:
        perturbations = specification_perturbations(self.baseline.design)
        self.assertEqual(len(perturbations), 36)
        self.assertEqual(len({item.parameter for item in perturbations}), 18)
        candidate = next(item for item in perturbations if item.parameter == "w3" and item.direction == "+")
        self.assertAlmostEqual(
            candidate.design.widths[2],
            self.baseline.design.widths[2] + 0.1,
        )
        np.testing.assert_array_equal(
            candidate.design.mirror_heights,
            self.baseline.design.mirror_heights,
        )

    def test_smoke_evaluation_accepts_zero_increment(self) -> None:
        evaluation = evaluate_design(
            baseline=self.baseline,
            design=self.baseline.design,
            profile=smoke_profile(),
            cache=EvaluationCache(),
        )
        self.assertEqual(evaluation.mirror_count, 1471)
        self.assertTrue(evaluation.geometry.valid)
        self.assertGreater(evaluation.annual_power_mw, 0.0)

    def test_search_trace_uses_explicit_six_group_reference(self) -> None:
        evaluation = evaluate_design(
            baseline=self.baseline,
            design=self.baseline.design,
            profile=smoke_profile(),
            cache=EvaluationCache(),
        )
        outcome = coordinate_search(
            initial_design=self.baseline.design,
            initial_evaluation=evaluation,
            active_variables=(),
            evaluator=lambda design: evaluation,
            baseline_q_kw_m2=0.5,
            maximum_sweeps=0,
        )
        self.assertEqual(outcome.trace, ())


if __name__ == "__main__":
    unittest.main()
