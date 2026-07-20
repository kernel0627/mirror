"""五节点 Campo 连续模型的三初值收敛搜索与最终验收。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .evaluate import (
    EvaluationCache,
    HeterogeneousEvaluation,
    dense_profile,
    evaluate_design,
    formal_profile,
    medium_profile,
    smoke_profile,
)
from .export import write_dense_validation, write_question3_results
from .model import (
    CampoMotherField,
    SplineDesign,
    build_campo_mother_field,
    fit_spline_design,
)
from .search import MultiStartOutcome, optimize_three_starts


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
DEFAULT_Q2_COORDINATES = (
    PROJECT_ROOT / "outputs" / "q2" / "03_最终镜位坐标.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3_continuous"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="求解第三问五节点径向连续 Campo 异构镜场"
    )
    parser.add_argument(
        "--q2-summary",
        type=Path,
        default=DEFAULT_Q2_SUMMARY,
    )
    parser.add_argument(
        "--q2-coordinates",
        type=Path,
        default=DEFAULT_Q2_COORDINATES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--target-power", type=float, default=42.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-joint-cycles",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-rounds-per-step",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--convergence-q",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--move-q",
        type=float,
        default=1e-7,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_power <= 0.0:
        raise SystemExit("--target-power 必须大于 0。")
    if args.workers < 1:
        raise SystemExit("--workers 必须大于等于 1。")
    if args.max_joint_cycles < 1:
        raise SystemExit("--max-joint-cycles 必须大于等于 1。")
    if args.max_rounds_per_step < 2:
        raise SystemExit("--max-rounds-per-step 必须大于等于 2。")
    if args.convergence_q <= 0.0 or args.move_q < 0.0:
        raise SystemExit("收敛阈值必须为正，移动阈值不能为负。")


def _previous_projection(
    mother: CampoMotherField,
    output_dir: Path,
) -> SplineDesign:
    candidates = (
        output_dir / "02_逐镜最终参数.csv",
        output_dir / "03_最终逐镜参数与坐标.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        widths: list[float] = []
        heights: list[float] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            z_column = (
                "installation_height_m"
                if "installation_height_m" in reader.fieldnames
                else "z_m"
            )
            if not {"mirror_width_m", z_column} <= set(
                reader.fieldnames
            ):
                continue
            for row in reader:
                widths.append(float(row["mirror_width_m"]))
                heights.append(float(row[z_column]))
        if len(widths) != mother.mirror_count:
            continue
        projected = fit_spline_design(
            mother,
            widths=widths,
            installation_heights=heights,
        )
        return SplineDesign(
            size_nodes=projected.size_nodes,
            height_nodes=projected.height_nodes,
            area_scale=1.0,
        ).canonical()
    return SplineDesign.uniform(mother.base_installation_height)


def _initial_designs(
    mother: CampoMotherField,
    previous: SplineDesign,
) -> tuple[tuple[str, SplineDesign], ...]:
    uniform = SplineDesign.uniform(mother.base_installation_height)
    weak = SplineDesign(
        size_nodes=tuple(-0.01 * (index - 2) for index in range(5)),
        height_nodes=tuple(
            mother.base_installation_height + 0.1 * (index - 2)
            for index in range(5)
        ),
        area_scale=1.0,
    ).canonical()
    return (
        ("uniform", uniform),
        ("previous_projection", previous),
        ("weak_engineering", weak),
    )


def _formal_selection(
    *,
    mother: CampoMotherField,
    search: MultiStartOutcome,
    target_power_mw: float,
) -> tuple[
    str,
    SplineDesign,
    HeterogeneousEvaluation,
    tuple[tuple[str, HeterogeneousEvaluation], ...],
]:
    cache = EvaluationCache()
    profile = formal_profile()
    candidates = tuple(
        (
            outcome.start_name,
            outcome.best_design,
            evaluate_design(
                mother=mother,
                design=outcome.best_design,
                profile=profile,
                cache=cache,
            ),
        )
        for outcome in search.starts
    )
    feasible = [
        candidate
        for candidate in candidates
        if candidate[2].is_feasible(target_power_mw)
    ]
    if not feasible:
        powers = ", ".join(
            f"{name}={evaluation.annual_power_mw:.6f}"
            for name, _, evaluation in candidates
        )
        raise RuntimeError(
            "三个收敛方案在正式精度下均不满足 42 MW："
            f"{powers}。"
        )
    selected = max(
        feasible,
        key=lambda item: item[2].unit_area_power_kw_m2,
    )
    return (
        selected[0],
        selected[1],
        selected[2],
        tuple((name, evaluation) for name, _, evaluation in candidates),
    )


def _clean_generated_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    mother = build_campo_mother_field(
        args.q2_summary,
        selected_coordinates_path=args.q2_coordinates,
    )
    previous = _previous_projection(mother, args.output)
    starts = _initial_designs(mother, previous)
    search_profile = smoke_profile() if args.smoke else medium_profile()
    maximum_joint_cycles = (
        min(args.max_joint_cycles, 2)
        if args.smoke
        else args.max_joint_cycles
    )

    print(
        "固定问题二 Campo 镜场："
        f"{mother.mirror_count} 面，"
        f"控制环={mother.control_ring_indices}，"
        f"控制半径={tuple(round(value, 3) for value in mother.control_radii)}",
        flush=True,
    )
    search = optimize_three_starts(
        mother=mother,
        initial_designs=starts,
        profile=search_profile,
        target_power_mw=args.target_power,
        move_q_threshold=args.move_q,
        convergence_q_threshold=args.convergence_q,
        maximum_joint_cycles=maximum_joint_cycles,
        maximum_rounds_per_step=args.max_rounds_per_step,
        workers=args.workers,
        progress=lambda message: print(message, flush=True),
    )

    if args.smoke:
        selected_start = search.best_start_name
        selected_design = search.best_design
        selected = search.best_evaluation
        formal_evaluations = tuple(
            (outcome.start_name, outcome.best_evaluation)
            for outcome in search.starts
        )
    else:
        (
            selected_start,
            selected_design,
            selected,
            formal_evaluations,
        ) = _formal_selection(
            mother=mother,
            search=search,
            target_power_mw=args.target_power,
        )

    _clean_generated_files(args.output)
    written = write_question3_results(
        output_dir=args.output,
        mother=mother,
        design=selected_design,
        evaluation=selected,
        search=search,
        selected_start_name=selected_start,
        formal_evaluations=formal_evaluations,
    )

    if not args.smoke:
        dense80 = dense_profile()
        dense100 = replace(
            dense80,
            name="q3-continuous-dense-100m",
            solver=replace(
                dense80.solver,
                neighbor_radius_m=100.0,
            ),
        )
        cache = EvaluationCache()
        evaluation80 = evaluate_design(
            mother=mother,
            design=selected_design,
            profile=dense80,
            cache=cache,
        )
        evaluation100 = evaluate_design(
            mother=mother,
            design=selected_design,
            profile=dense100,
            cache=cache,
        )
        written["dense_validation"] = write_dense_validation(
            output_dir=args.output,
            evaluations=(
                (dense80, evaluation80),
                (dense100, evaluation100),
            ),
        )

    print("\n五节点 Campo 连续模型结果", flush=True)
    print(f"正式起点：{selected_start}", flush=True)
    print(f"镜子数：{selected.mirror_count}", flush=True)
    print(
        f"总镜面面积：{selected.total_area_m2:.3f} m²",
        flush=True,
    )
    print(
        f"年平均输出热功率：{selected.annual_power_mw:.6f} MW",
        flush=True,
    )
    print(
        "单位镜面面积年平均输出："
        f"{selected.unit_area_power_kw_m2:.6f} kW/m²",
        flush=True,
    )
    for outcome in search.starts:
        print(
            f"{outcome.start_name}："
            f"{outcome.stopped_by}，"
            f"联合循环 {outcome.joint_cycles}，"
            f"稳定轮数 {outcome.stable_joint_cycles}",
            flush=True,
        )
    for path in written.values():
        print(f"输出：{path}", flush=True)
    return 0


def main() -> None:
    raise SystemExit(run())
