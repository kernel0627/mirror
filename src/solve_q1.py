#!/usr/bin/env python3
"""运行 A 题第一问的正式光学效率计算。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from heliostat.config import FieldConfig, SolverConfig
from heliostat.evaluator import TimeResult, solve_question1
from heliostat.geometry import prepare_field
from heliostat.io import load_mirror_xy, write_question1_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "task" / "A" / "fj.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q1"


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("月份应使用逗号分隔的整数。") from exc
    if not result:
        raise argparse.ArgumentTypeError("月份列表不能为空。")
    return result


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("时刻应使用逗号分隔的数字。") from exc
    if not result:
        raise argparse.ArgumentTypeError("时刻列表不能为空。")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算 CUMCM 2023 A 题第一问的镜场光学效率和输出热功率"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shadow-grid", type=int, default=15)
    parser.add_argument("--truncation-rays", type=int, default=256)
    parser.add_argument("--neighbor-radius", type=float, default=60.0)
    parser.add_argument("--truncation-chunk-size", type=int, default=128)
    parser.add_argument("--sobol-seed", type=int, default=2023)
    parser.add_argument(
        "--months",
        type=_comma_ints,
        default=tuple(range(1, 13)),
        help="逗号分隔；默认 1 到 12 月",
    )
    parser.add_argument(
        "--times",
        type=_comma_floats,
        default=(9.0, 10.5, 12.0, 13.5, 15.0),
        help="逗号分隔的当地太阳时",
    )
    parser.add_argument(
        "--limit-mirrors",
        type=int,
        default=None,
        help="仅用于调试；只计算附件中的前 N 面镜子",
    )
    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--skip-truncation", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _progress(current: int, total: int, record: TimeResult) -> None:
    hour = int(record.solar_time)
    minute = int(round((record.solar_time - hour) * 60.0))
    print(
        f"[{current:02d}/{total:02d}] "
        f"{record.month:02d}月21日 {hour:02d}:{minute:02d} "
        f"光学效率={record.average_optical_efficiency:.4f} "
        f"输出={record.field_output_mw:.3f} MW"
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mirror_xy = load_mirror_xy(args.input)
    if args.limit_mirrors is not None:
        if args.limit_mirrors < 1:
            raise SystemExit("--limit-mirrors 必须大于等于 1。")
        mirror_xy = mirror_xy[: args.limit_mirrors]

    field_config = FieldConfig()
    solver_config = SolverConfig(
        shadow_grid_size=args.shadow_grid,
        truncation_rays=args.truncation_rays,
        neighbor_radius_m=args.neighbor_radius,
        truncation_chunk_size=args.truncation_chunk_size,
        sobol_seed=args.sobol_seed,
        calculate_shadow=not args.skip_shadow,
        calculate_truncation=not args.skip_truncation,
    )
    prepared = prepare_field(mirror_xy, field_config)
    solution = solve_question1(
        prepared=prepared,
        solver=solver_config,
        months=args.months,
        solar_times=args.times,
        progress=None if args.quiet else _progress,
    )
    written = write_question1_results(
        output_dir=args.output,
        time_records=solution.time_results,
        monthly_records=solution.monthly_results,
        annual_record=solution.annual_result,
        field_config=field_config,
        solver_config=solver_config,
        source_path=args.input,
        mirror_count=prepared.mirror_count,
    )

    annual = solution.annual_result
    print("\n汇总结果")
    print(f"平均光学效率：{annual.average_optical_efficiency:.6f}")
    print(f"平均余弦效率：{annual.average_cosine_efficiency:.6f}")
    print(
        "平均阴影遮挡效率："
        f"{annual.average_shadow_blocking_efficiency:.6f}"
    )
    print(f"平均截断效率：{annual.average_truncation_efficiency:.6f}")
    print(f"平均输出热功率：{annual.field_output_mw:.6f} MW")
    print(
        "单位镜面面积平均输出热功率："
        f"{annual.unit_area_output_kw_m2:.6f} kW/m²"
    )
    print(f"结果目录：{args.output.resolve()}")
    for name, path in written.items():
        print(f"  {name}: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
