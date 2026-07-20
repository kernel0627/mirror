from __future__ import annotations

import unittest

import numpy as np

from heliostat.q3_continuous.evaluate import (
    EvaluationCache,
    evaluate_design,
    smoke_profile,
)
from heliostat.q3_continuous.model import (
    SplineDesign,
    build_campo_mother_field,
    expand_spline_design,
    fit_spline_design,
    validate_heterogeneous_field,
)
from heliostat.q3_continuous.search import (
    height_candidates,
    size_candidates,
)


class FiveNodeCampoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mother = build_campo_mother_field(
            "outputs/q2/07_最终方案摘要.json",
            selected_coordinates_path="outputs/q2/03_最终镜位坐标.csv",
        )

    def test_automatic_nodes_follow_campo_metadata(self) -> None:
        self.assertEqual(self.mother.mirror_count, 1469)
        self.assertEqual(self.mother.ring_count, 28)
        self.assertEqual(
            self.mother.control_ring_indices,
            (1, 4, 12, 14, 28),
        )
        self.assertTrue(
            all(
                left < right
                for left, right in zip(
                    self.mother.control_radii[:-1],
                    self.mother.control_radii[1:],
                )
            )
        )

    def test_basis_is_local_nonnegative_partition_of_unity(self) -> None:
        basis = self.mother.radial_basis
        self.assertEqual(basis.shape, (1469, 5))
        self.assertTrue(np.all(basis >= -1e-12))
        np.testing.assert_allclose(
            np.sum(basis, axis=1),
            1.0,
            atol=1e-12,
        )
        self.assertTrue(np.all(np.count_nonzero(basis > 1e-12, axis=1) <= 2))
        for node, radius in enumerate(self.mother.control_radii):
            active = np.isclose(self.mother.ring_radii, radius)
            np.testing.assert_allclose(basis[active, node], 1.0)

    def test_uniform_design_reproduces_q2_specifications(self) -> None:
        design = SplineDesign.uniform(
            self.mother.base_installation_height
        )
        specifications = expand_spline_design(self.mother, design)
        np.testing.assert_allclose(
            specifications.widths,
            self.mother.base_width,
        )
        np.testing.assert_allclose(
            specifications.heights,
            self.mother.base_height,
        )
        np.testing.assert_allclose(
            specifications.installation_heights,
            self.mother.base_installation_height,
        )

    def test_size_shape_is_invariant_to_common_node_shift(self) -> None:
        first = SplineDesign(
            size_nodes=(-0.04, 0.03, 0.01, -0.02, 0.02),
            height_nodes=(3.0, 3.2, 3.8, 4.4, 5.0),
            area_scale=0.98,
        )
        second = SplineDesign(
            size_nodes=tuple(value + 0.17 for value in first.size_nodes),
            height_nodes=first.height_nodes,
            area_scale=first.area_scale,
        )
        left = expand_spline_design(self.mother, first)
        right = expand_spline_design(self.mother, second)
        np.testing.assert_allclose(left.widths, right.widths, atol=1e-12)
        np.testing.assert_allclose(left.heights, right.heights, atol=1e-12)

    def test_existing_spline_can_be_projected_back_to_nodes(self) -> None:
        design = SplineDesign(
            size_nodes=(-0.03, 0.02, 0.04, -0.01, -0.02),
            height_nodes=(3.0, 3.3, 4.1, 4.8, 5.3),
            area_scale=0.985,
        ).canonical()
        specifications = expand_spline_design(self.mother, design)
        projected = fit_spline_design(
            self.mother,
            widths=specifications.widths,
            installation_heights=specifications.installation_heights,
        )
        rebuilt = expand_spline_design(self.mother, projected)
        np.testing.assert_allclose(
            rebuilt.widths,
            specifications.widths,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            rebuilt.installation_heights,
            specifications.installation_heights,
            atol=1e-10,
        )

    def test_conservative_spline_passes_geometry(self) -> None:
        design = SplineDesign(
            size_nodes=(0.01, 0.0, -0.01, -0.02, -0.03),
            height_nodes=(3.4, 3.5, 4.0, 4.5, 5.0),
            area_scale=0.96,
        )
        specifications = expand_spline_design(self.mother, design)
        check = validate_heterogeneous_field(
            coordinates=self.mother.coordinates,
            widths=specifications.widths,
            heights=specifications.heights,
            installation_heights=specifications.installation_heights,
            tower_x=self.mother.parameters.tower_x,
            tower_y=self.mother.parameters.tower_y,
            safety_epsilon=self.mother.parameters.safety_epsilon,
        )
        self.assertTrue(check.valid, check.reason)

    def test_search_actions_include_required_move_types(self) -> None:
        design = SplineDesign.uniform(
            self.mother.base_installation_height
        )
        heights = height_candidates(design, 0.1)
        sizes = size_candidates(design, 0.01)
        self.assertEqual(len(heights), 18)
        self.assertEqual(len(sizes), 30)
        actions = [action for action, _ in sizes]
        self.assertTrue(any(action.startswith("adjacent-") for action in actions))
        self.assertTrue(any(action.startswith("cross-") for action in actions))


class FiveNodeCampoEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mother = build_campo_mother_field(
            "outputs/q2/07_最终方案摘要.json",
            selected_coordinates_path="outputs/q2/03_最终镜位坐标.csv",
        )

    def test_smoke_evaluation_uses_fixed_q2_field(self) -> None:
        evaluation = evaluate_design(
            mother=self.mother,
            design=SplineDesign.uniform(
                self.mother.base_installation_height
            ),
            profile=smoke_profile(),
            cache=EvaluationCache(),
        )
        self.assertEqual(evaluation.mirror_count, 1469)
        self.assertGreater(evaluation.annual_power_mw, 0.0)
        self.assertTrue(evaluation.geometry.valid)


if __name__ == "__main__":
    unittest.main()
