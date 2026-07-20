from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from heliostat.q3.evaluate import (
    EvaluationCache,
    evaluate_design,
    smoke_profile,
)
from heliostat.q3.export import (
    write_dense_validation,
    write_result3_workbook,
)
from heliostat.q3.model import (
    EXPECTED_GROUP_COUNTS,
    GroupDesign,
    build_campo_mother_field,
    expand_group_design,
    group_width_caps,
    validate_heterogeneous_field,
)
from heliostat.q3.search import transfer_area


class Question3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mother = build_campo_mother_field(
            "outputs/q2/07_最终方案摘要.json"
        )

    def test_mother_field_has_recorded_group_structure(self) -> None:
        self.assertEqual(self.mother.mirror_count, 1471)
        self.assertEqual(
            self.mother.group_counts,
            EXPECTED_GROUP_COUNTS,
        )
        np.testing.assert_array_equal(
            np.unique(self.mother.group_indices),
            np.arange(6),
        )

    def test_uniform_design_is_geometrically_legal(self) -> None:
        design = GroupDesign.uniform(
            self.mother.base_installation_height
        )
        specs = expand_group_design(self.mother, design)
        check = validate_heterogeneous_field(
            coordinates=self.mother.coordinates,
            widths=specs.widths,
            heights=specs.heights,
            installation_heights=specs.installation_heights,
            tower_x=self.mother.parameters.tower_x,
            tower_y=self.mother.parameters.tower_y,
            safety_epsilon=self.mother.parameters.safety_epsilon,
        )

        self.assertTrue(check.valid, check.reason)
        self.assertGreaterEqual(
            check.minimum_width_clearance_m,
            0.01 - 1e-9,
        )
        caps = group_width_caps(self.mother)
        self.assertAlmostEqual(
            caps[0],
            self.mother.base_width,
            places=9,
        )
        self.assertGreater(caps[2], 8.0 - 1e-9)

    def test_area_transfer_conserves_total_area(self) -> None:
        design = GroupDesign.uniform(
            self.mother.base_installation_height
        )
        base_area = self.mother.base_width * self.mother.base_height
        transferred = transfer_area(
            design=design,
            source_group=3,
            target_group=2,
            delta_area_m2=100.0,
            group_counts=self.mother.group_counts,
            base_mirror_area_m2=base_area,
        )
        before = sum(
            count * base_area * scale**2
            for count, scale in zip(
                self.mother.group_counts,
                design.scales,
            )
        )
        after = sum(
            count * base_area * scale**2
            for count, scale in zip(
                self.mother.group_counts,
                transferred.scales,
            )
        )
        self.assertAlmostEqual(after, before, places=9)


class Question3EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mother = build_campo_mother_field(
            "outputs/q2/07_最终方案摘要.json"
        )
        cls.design = GroupDesign.uniform(
            cls.mother.base_installation_height
        )
        cls.evaluation = evaluate_design(
            mother=cls.mother,
            design=cls.design,
            profile=smoke_profile(),
            cache=EvaluationCache(),
        )

    def test_smoke_evaluation_uses_all_mirrors(self) -> None:
        self.assertEqual(self.evaluation.mirror_count, 1471)
        self.assertGreater(self.evaluation.annual_power_mw, 0.0)
        self.assertGreater(
            self.evaluation.unit_area_power_kw_m2,
            0.0,
        )
        self.assertTrue(self.evaluation.geometry.valid)

    def test_result3_export_contains_individual_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result3.xlsx"
            write_result3_workbook(
                template_path="task/A/result3.xlsx",
                output_path=output,
                evaluation=self.evaluation,
                tower_x=self.mother.parameters.tower_x,
                tower_y=self.mother.parameters.tower_y,
            )
            workbook = load_workbook(
                output,
                read_only=True,
                data_only=True,
            )
            sheet = workbook.active
            self.assertEqual(
                sheet.max_row,
                self.evaluation.mirror_count + 1,
            )
            self.assertAlmostEqual(
                sheet.cell(2, 4).value,
                float(self.evaluation.widths[0]),
            )
            self.assertAlmostEqual(
                sheet.cell(2, 5).value,
                float(self.evaluation.heights[0]),
            )
            self.assertAlmostEqual(
                sheet.cell(2, 8).value,
                float(self.evaluation.installation_heights[0]),
            )
            workbook.close()

    def test_dense_validation_records_neighbor_sensitivity_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            table_path = output / "08_论文结果与验证表.md"
            table_path.write_text("# 第三问结果\n", encoding="utf-8")
            profile_80 = smoke_profile()
            profile_100 = replace(
                profile_80,
                name="smoke-100m",
                solver=replace(
                    profile_80.solver,
                    neighbor_radius_m=100.0,
                ),
            )
            arguments = {
                "output_dir": output,
                "evaluation": self.evaluation,
                "profile": profile_80,
                "sensitivity_evaluations": (
                    (profile_100, self.evaluation),
                ),
            }
            write_dense_validation(**arguments)
            write_dense_validation(**arguments)

            payload = json.loads(
                (output / "09_高精度加密验证.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                len(payload["neighbor_radius_sensitivity"]),
                2,
            )
            table = table_path.read_text(encoding="utf-8")
            self.assertEqual(table.count("## 表 6 "), 1)
            self.assertIn("| 100 ", table)


if __name__ == "__main__":
    unittest.main()
