"""第一问逐时刻计算、验证运行和命令行入口。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..config import FieldConfig, SolverConfig
from ..geometry import (
    PreparedField,
    calculate_orientation,
    maximum_reflection_error,
    prepare_field,
)
from ..io import load_mirror_xy
from ..shadow import calculate_shadow_blocking_efficiency
from ..solar import calculate_solar_state
from ..truncation import calculate_truncation_efficiency
from .aggregate import (
    Question1Solution,
    summarize_annual,
    summarize_mirror_annual,
    summarize_monthly,
)
from .export import (
    write_paper_tables,
    write_question1_results,
    write_validation_table,
)


SOLAR_TIMES = (9.0, 10.5, 12.0, 13.5, 15.0)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "task" / "A" / "fj.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q1"


@dataclass(frozen=True)
class TimeResult:
    month: int
    solar_time: float
    dni_kw_m2: float
    average_optical_efficiency: float
    average_cosine_efficiency: float
    average_shadow_blocking_efficiency: float
    average_atmospheric_efficiency: float
    average_truncation_efficiency: float
    field_output_mw: float
    unit_area_output_kw_m2: float
    maximum_reflection_error: float


@dataclass(frozen=True)
class ValidationResult:
    category: str
    parameter: str
    metric: str
    value: float
    relative_difference_percent: float
    runtime_seconds: float


ProgressCallback = Callable[[int, int, TimeResult], None]


def _check_efficiency(name: str, values: np.ndarray) -> None:
    tolerance = 1e-12
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        raise RuntimeError(
            f"{name} 超出 [0, 1]：min={minimum:.6g}, max={maximum:.6g}"
        )


def evaluate_time(
    prepared: PreparedField,
    month: int,
    solar_time: float,
    solver: SolverConfig,
    mirror_sums: dict[str, np.ndarray] | None = None,
) -> TimeResult:
    """计算一个月份、一个规定时刻的全场平均结果。"""

    solar = calculate_solar_state(
        month=month,
        solar_time=solar_time,
        latitude_deg=prepared.config.latitude_deg,
        altitude_km=prepared.config.altitude_km,
    )
    orientation = calculate_orientation(prepared, solar.direction)
    reflection_error = maximum_reflection_error(
        prepared,
        orientation,
        solar.direction,
    )
    if reflection_error >= 1e-8:
        raise RuntimeError(f"中心光线反射误差过大：{reflection_error:.3e}")

    if solver.calculate_shadow:
        shadow = calculate_shadow_blocking_efficiency(
            prepared,
            orientation,
            solar.direction,
            solver,
        )
    else:
        shadow = np.ones(prepared.mirror_count, dtype=float)

    if solver.calculate_truncation:
        truncation = calculate_truncation_efficiency(
            prepared,
            orientation,
            solar.direction,
            solver,
        )
    else:
        truncation = np.ones(prepared.mirror_count, dtype=float)

    cosine = orientation.cosine_efficiency
    atmospheric = prepared.atmospheric_efficiency
    optical = (
        cosine
        * shadow
        * atmospheric
        * truncation
        * prepared.config.reflectivity
    )
    for name, values in (
        ("余弦效率", cosine),
        ("阴影遮挡效率", shadow),
        ("大气透射率", atmospheric),
        ("截断效率", truncation),
        ("光学效率", optical),
    ):
        _check_efficiency(name, values)

    mirror_power_kw = solar.dni_kw_m2 * prepared.config.mirror_area * optical
    if mirror_sums is not None:
        mirror_sums["optical_efficiency_sum"] += optical
        mirror_sums["cosine_efficiency_sum"] += cosine
        mirror_sums["shadow_blocking_efficiency_sum"] += shadow
        mirror_sums["atmospheric_efficiency_sum"] += atmospheric
        mirror_sums["truncation_efficiency_sum"] += truncation
        mirror_sums["output_power_kw_sum"] += mirror_power_kw

    field_power_kw = float(np.sum(mirror_power_kw))
    return TimeResult(
        month=month,
        solar_time=solar_time,
        dni_kw_m2=solar.dni_kw_m2,
        average_optical_efficiency=float(np.mean(optical)),
        average_cosine_efficiency=float(np.mean(cosine)),
        average_shadow_blocking_efficiency=float(np.mean(shadow)),
        average_atmospheric_efficiency=float(np.mean(atmospheric)),
        average_truncation_efficiency=float(np.mean(truncation)),
        field_output_mw=field_power_kw / 1000.0,
        unit_area_output_kw_m2=field_power_kw / prepared.total_mirror_area,
        maximum_reflection_error=reflection_error,
    )


def solve_question1(
    prepared: PreparedField,
    solver: SolverConfig,
    months: Sequence[int] = tuple(range(1, 13)),
    solar_times: Sequence[float] = SOLAR_TIMES,
    progress: ProgressCallback | None = None,
) -> Question1Solution:
    """执行所选月份和时刻；默认即题目规定的 60 个状态。"""

    if not months or not solar_times:
        raise ValueError("months 和 solar_times 不能为空。")
    if any(month < 1 or month > 12 for month in months):
        raise ValueError("months 必须位于 1 到 12。")

    records: list[TimeResult] = []
    mirror_sums = {
        name: np.zeros(prepared.mirror_count, dtype=float)
        for name in (
            "optical_efficiency_sum",
            "cosine_efficiency_sum",
            "shadow_blocking_efficiency_sum",
            "atmospheric_efficiency_sum",
            "truncation_efficiency_sum",
            "output_power_kw_sum",
        )
    }
    total = len(months) * len(solar_times)
    for month in months:
        for solar_time in solar_times:
            record = evaluate_time(
                prepared,
                month,
                solar_time,
                solver,
                mirror_sums=mirror_sums,
            )
            records.append(record)
            if progress is not None:
                progress(len(records), total, record)

    time_results = tuple(records)
    return Question1Solution(
        time_results=time_results,
        monthly_results=summarize_monthly(time_results),
        annual_result=summarize_annual(time_results),
        mirror_annual_results=summarize_mirror_annual(
            mirror_xy=prepared.centers[:, :2],
            tower_x=prepared.config.tower_x,
            tower_y=prepared.config.tower_y,
            state_count=len(time_results),
            **mirror_sums,
        ),
    )


def run_validation_suite(
    prepared: PreparedField,
    base_solver: SolverConfig,
) -> tuple[ValidationResult, ...]:
    """运行三组隔离后的收敛实验，供一张验证表使用。"""

    specifications = [
        (
            "阴影网格",
            "10×10",
            replace(
                base_solver,
                shadow_grid_size=10,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            False,
        ),
        (
            "阴影网格",
            "15×15",
            replace(
                base_solver,
                shadow_grid_size=15,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            True,
        ),
        (
            "阴影网格",
            "20×20",
            replace(
                base_solver,
                shadow_grid_size=20,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            False,
        ),
        (
            "邻镜半径",
            "40 m",
            replace(
                base_solver,
                neighbor_radius_m=40.0,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            False,
        ),
        (
            "邻镜半径",
            "60 m",
            replace(
                base_solver,
                neighbor_radius_m=60.0,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            True,
        ),
        (
            "邻镜半径",
            "80 m",
            replace(
                base_solver,
                neighbor_radius_m=80.0,
                calculate_shadow=True,
                calculate_truncation=False,
            ),
            "average_shadow_blocking_efficiency",
            "年平均阴影遮挡效率",
            False,
        ),
        (
            "截断光线",
            "128",
            replace(
                base_solver,
                truncation_rays=128,
                calculate_shadow=False,
                calculate_truncation=True,
            ),
            "average_truncation_efficiency",
            "年平均截断效率",
            False,
        ),
        (
            "截断光线",
            "256",
            replace(
                base_solver,
                truncation_rays=256,
                calculate_shadow=False,
                calculate_truncation=True,
            ),
            "average_truncation_efficiency",
            "年平均截断效率",
            True,
        ),
        (
            "截断光线",
            "512",
            replace(
                base_solver,
                truncation_rays=512,
                calculate_shadow=False,
                calculate_truncation=True,
            ),
            "average_truncation_efficiency",
            "年平均截断效率",
            False,
        ),
    ]

    raw: list[tuple[str, str, str, float, float, bool]] = []
    for category, parameter, solver, field, metric, reference in specifications:
        started = time.perf_counter()
        solution = solve_question1(prepared, solver)
        elapsed = time.perf_counter() - started
        value = float(getattr(solution.annual_result, field))
        raw.append((category, parameter, metric, value, elapsed, reference))

    baselines = {
        category: value
        for category, _, _, value, _, reference in raw
        if reference
    }
    return tuple(
        ValidationResult(
            category=category,
            parameter=parameter,
            metric=metric,
            value=value,
            relative_difference_percent=(
                abs(value - baselines[category]) / abs(baselines[category]) * 100.0
            ),
            runtime_seconds=elapsed,
        )
        for category, parameter, metric, value, elapsed, _ in raw
    )


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
        default=SOLAR_TIMES,
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
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="额外运行三组收敛实验并生成一张验证表",
    )
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
        mirror_annual_records=solution.mirror_annual_results,
        field_config=field_config,
        solver_config=solver_config,
        source_path=args.input,
        mirror_count=prepared.mirror_count,
    )
    written.update(
        write_paper_tables(
            args.output,
            solution.monthly_results,
            solution.annual_result,
        )
    )

    if args.run_validation:
        validation = run_validation_suite(prepared, solver_config)
        written.update(write_validation_table(args.output, validation))

    if not args.skip_figures:
        from .plot import build_paper_figures

        written.update(
            build_paper_figures(
                output_dir=args.output,
            )
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
        print(f"  {name}: {path.relative_to(args.output)}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
