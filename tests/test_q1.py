from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heliostat.config import FieldConfig, SolverConfig
from heliostat.geometry import prepare_field
from heliostat.q1.aggregate import (
    MirrorAnnualResult,
    summarize_annual,
    summarize_monthly,
)
from heliostat.q1.export import write_paper_tables, write_question1_results
from heliostat.q1.solve import TimeResult, solve_question1


def _time_result(month: int, solar_time: float, value: float) -> TimeResult:
    return TimeResult(
        month=month,
        solar_time=solar_time,
        dni_kw_m2=value,
        average_optical_efficiency=value,
        average_cosine_efficiency=value,
        average_shadow_blocking_efficiency=value,
        average_atmospheric_efficiency=value,
        average_truncation_efficiency=value,
        field_output_mw=value,
        unit_area_output_kw_m2=value,
        maximum_reflection_error=0.0,
    )


class Question1AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            _time_result(1, 9.0, 0.2),
            _time_result(1, 12.0, 0.4),
            _time_result(2, 9.0, 0.6),
            _time_result(2, 12.0, 0.8),
        )

    def test_monthly_and_annual_means(self) -> None:
        monthly = summarize_monthly(self.records)
        annual = summarize_annual(self.records)

        self.assertEqual([record.month for record in monthly], [1, 2])
        self.assertAlmostEqual(monthly[0].average_optical_efficiency, 0.3)
        self.assertAlmostEqual(monthly[1].field_output_mw, 0.7)
        self.assertAlmostEqual(annual.average_optical_efficiency, 0.5)

    def test_exports_mirror_annual_without_mirror_time_detail(self) -> None:
        monthly = summarize_monthly(self.records)
        annual = summarize_annual(self.records)
        mirror_annual = (
            MirrorAnnualResult(
                mirror_id=1,
                x_m=120.0,
                y_m=0.0,
                radius_to_tower_m=120.0,
                average_optical_efficiency=0.5,
                average_cosine_efficiency=0.6,
                average_shadow_blocking_efficiency=0.9,
                average_atmospheric_efficiency=0.98,
                average_truncation_efficiency=0.99,
                average_output_power_kw=25.0,
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            written = write_question1_results(
                output_dir=output,
                time_records=self.records,
                monthly_records=monthly,
                annual_record=annual,
                mirror_annual_records=mirror_annual,
                field_config=FieldConfig(),
                solver_config=SolverConfig(),
                source_path="task/A/fj.xlsx",
                mirror_count=1745,
            )
            written.update(write_paper_tables(output, monthly, annual))

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "02_逐时刻计算结果.csv",
                    "03_月平均计算结果.csv",
                    "04_年平均计算结果.json",
                    "05_单镜年平均结果.csv",
                    "06_运行配置.json",
                    "07_论文结果与验证表.md",
                },
            )
            self.assertFalse(
                any("单镜逐时刻" in path.name for path in output.rglob("*"))
            )
            run_config = json.loads(
                (output / "06_运行配置.json").read_text()
            )
            self.assertEqual(run_config["source"], "task/A/fj.xlsx")
            self.assertEqual(run_config["run"]["time_state_count"], 4)
            self.assertEqual(
                len(
                    (output / "05_单镜年平均结果.csv")
                    .read_text(encoding="utf-8-sig")
                    .splitlines()
                ),
                2,
            )
            self.assertTrue(all(path.exists() for path in written.values()))

    def test_mirror_annual_aggregation_matches_field_annual(self) -> None:
        prepared = prepare_field(
            np.array([[120.0, 0.0], [140.0, 20.0]], dtype=float),
            FieldConfig(),
        )
        solution = solve_question1(
            prepared,
            SolverConfig(
                shadow_grid_size=3,
                truncation_rays=8,
                calculate_shadow=False,
                calculate_truncation=False,
            ),
            months=(6,),
            solar_times=(9.0, 12.0),
        )

        self.assertEqual(len(solution.mirror_annual_results), 2)
        self.assertAlmostEqual(
            np.mean(
                [
                    record.average_optical_efficiency
                    for record in solution.mirror_annual_results
                ]
            ),
            solution.annual_result.average_optical_efficiency,
        )
        self.assertAlmostEqual(
            sum(
                record.average_output_power_kw
                for record in solution.mirror_annual_results
            )
            / 1000.0,
            solution.annual_result.field_output_mw,
        )


if __name__ == "__main__":
    unittest.main()
