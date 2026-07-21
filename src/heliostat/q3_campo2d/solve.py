"""第三问 Campo2D 的多起点搜索、正式验收与结果导出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evaluate import (
    Campo2DEvaluation,
    EvaluationCache,
    coarse_profile,
    dense_profile,
    evaluate_design,
    formal_profile,
    medium_profile,
    smoke_profile,
)
from .export import write_dense_validation, write_primary_results
from .model import Campo2DBase, Campo2DDesign, load_q2_campo_base
from .plot import generate_figures
from .search import MultiStartOutcome, optimize_multi_start
from ..q2.evaluate import EvaluationProfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
DEFAULT_Q2_COORDINATES = PROJECT_ROOT / "outputs" / "q2" / "03_最终镜位坐标.csv"
DEFAULT_Q2_MONTHLY = PROJECT_ROOT / "outputs" / "q2" / "04_月平均计算结果.csv"
DEFAULT_Q2_MIRRORS = PROJECT_ROOT / "outputs" / "q2" / "06_单镜年平均结果.csv"
DEFAULT_SIX_SUMMARY = PROJECT_ROOT / "outputs" / "q3" / "07_最终方案摘要.json"
DEFAULT_TEMPLATE = PROJECT_ROOT / "task" / "A" / "result3.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3_campo2d"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="求解第三问径向—角度连续异构 Campo 镜场"
    )
    parser.add_argument("--q2-summary", type=Path, default=DEFAULT_Q2_SUMMARY)
    parser.add_argument("--q2-coordinates", type=Path, default=DEFAULT_Q2_COORDINATES)
    parser.add_argument("--q2-monthly", type=Path, default=DEFAULT_Q2_MONTHLY)
    parser.add_argument("--q2-mirrors", type=Path, default=DEFAULT_Q2_MIRRORS)
    parser.add_argument("--six-summary", type=Path, default=DEFAULT_SIX_SUMMARY)
    parser.add_argument("--result3-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sobol-count", type=int, default=16)
    parser.add_argument("--retained-starts", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-joint-cycles", type=int, default=6)
    parser.add_argument("--medium-candidates", type=int, default=4)
    parser.add_argument("--target-power", type=float, default=42.0)
    parser.add_argument("--move-q", type=float, default=1e-5)
    parser.add_argument("--convergence-q", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2023)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.sobol_count < 0:
        raise SystemExit("--sobol-count 不能小于 0。")
    if args.retained_starts < 1:
        raise SystemExit("--retained-starts 必须大于等于 1。")
    if args.max_rounds < 1 or args.max_joint_cycles < 1:
        raise SystemExit("搜索轮数必须大于等于 1。")
    if args.medium_candidates < 1:
        raise SystemExit("--medium-candidates 必须大于等于 1。")
    if args.target_power <= 0.0:
        raise SystemExit("--target-power 必须大于 0。")
    if args.move_q < 0.0 or args.convergence_q <= 0.0:
        raise SystemExit("搜索阈值不合法。")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象。")
    return payload


def _verify_q2_uniform_regression(
    *,
    base: Campo2DBase,
    q2_summary: dict,
) -> Campo2DEvaluation:
    """正式搜索前验证新异构评价器能复现问题二统一规格结果。"""

    design = Campo2DDesign.uniform(
        base.parameters,
        ring_count=base.ring_count,
    )
    evaluation = evaluate_design(
        base=base,
        design=design,
        profile=formal_profile(),
        cache=EvaluationCache(),
    )
    annual = q2_summary.get("annual", {})
    expected_power = float(annual["field_output_mw"])
    expected_q = float(annual["unit_area_output_kw_m2"])
    expected_area = float(q2_summary["total_area_m2"])
    expected_count = int(q2_summary["mirror_count"])
    errors = {
        "mirror_count": abs(evaluation.mirror_count - expected_count),
        "total_area_m2": abs(evaluation.total_area_m2 - expected_area),
        "annual_power_mw": abs(evaluation.annual_power_mw - expected_power),
        "unit_area_power_kw_m2": abs(
            evaluation.unit_area_power_kw_m2 - expected_q
        ),
    }
    tolerances = {
        "mirror_count": 0,
        "total_area_m2": 1e-6,
        "annual_power_mw": 1e-6,
        "unit_area_power_kw_m2": 1e-9,
    }
    failures = [
        f"{name} 误差={errors[name]:.12g}"
        for name in errors
        if errors[name] > tolerances[name]
    ]
    if failures:
        raise RuntimeError("问题二统一规格正式回归失败：" + "; ".join(failures))
    return evaluation


def _formal_candidates(
    *,
    base: Campo2DBase,
    search: MultiStartOutcome,
    target_power_mw: float,
    smoke: bool,
) -> tuple[
    str,
    Campo2DDesign,
    Campo2DEvaluation,
    tuple[tuple[str, Campo2DEvaluation], ...],
]:
    profile = smoke_profile() if smoke else formal_profile()
    cache = EvaluationCache()
    candidates = tuple(
        (
            outcome.start_name,
            outcome.best_design,
            evaluate_design(
                base=base,
                design=outcome.best_design,
                profile=profile,
                cache=cache,
            ),
        )
        for outcome in search.starts
    )
    feasible = [item for item in candidates if item[2].is_feasible(target_power_mw)]
    if not feasible:
        powers = ", ".join(
            f"{name}={evaluation.annual_power_mw:.6f}"
            for name, _, evaluation in candidates
        )
        raise RuntimeError(f"正式候选均不满足功率约束：{powers} MW。")
    selected = max(feasible, key=lambda item: item[2].unit_area_power_kw_m2)
    return (
        selected[0],
        selected[1],
        selected[2],
        tuple((name, evaluation) for name, _, evaluation in candidates),
    )


def _select_dense_feasible_candidate(
    *,
    base: Campo2DBase,
    formal_candidates: tuple[tuple[str, Campo2DEvaluation], ...],
    target_power_mw: float,
) -> tuple[
    str,
    Campo2DDesign,
    Campo2DEvaluation,
    tuple[
        tuple[EvaluationProfile, Campo2DEvaluation],
        tuple[EvaluationProfile, Campo2DEvaluation],
    ],
]:
    """按正式 q 排序，并以两档加密复算作为最终可行性门槛。"""

    dense80 = dense_profile(neighbor_radius_m=80.0)
    dense100 = dense_profile(neighbor_radius_m=100.0)
    cache = EvaluationCache()
    ordered = sorted(
        (
            (name, evaluation)
            for name, evaluation in formal_candidates
            if evaluation.is_feasible(target_power_mw)
        ),
        key=lambda item: item[1].unit_area_power_kw_m2,
        reverse=True,
    )
    failures: list[str] = []
    for name, formal in ordered:
        evaluation80 = evaluate_design(
            base=base,
            design=formal.design,
            profile=dense80,
            cache=cache,
        )
        evaluation100 = evaluate_design(
            base=base,
            design=formal.design,
            profile=dense100,
            cache=cache,
        )
        if evaluation80.is_feasible(target_power_mw) and evaluation100.is_feasible(
            target_power_mw
        ):
            return (
                name,
                formal.design,
                formal,
                ((dense80, evaluation80), (dense100, evaluation100)),
            )
        failures.append(
            f"{name}: 80m={evaluation80.annual_power_mw:.6f}, "
            f"100m={evaluation100.annual_power_mw:.6f} MW"
        )
    detail = "; ".join(failures) if failures else "无正式可行候选"
    raise RuntimeError(f"没有候选同时通过两档加密功率约束：{detail}。")


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    q2_summary = _load_json(args.q2_summary)
    six_summary = _load_json(args.six_summary)
    base = load_q2_campo_base(args.q2_summary, args.q2_coordinates)
    search_coarse = smoke_profile() if args.smoke else coarse_profile()
    search_medium = smoke_profile() if args.smoke else medium_profile()
    sobol_count = min(args.sobol_count, 1) if args.smoke else args.sobol_count
    retained_starts = min(args.retained_starts, 2) if args.smoke else args.retained_starts
    maximum_rounds = min(args.max_rounds, 1) if args.smoke else args.max_rounds
    maximum_joint_cycles = (
        min(args.max_joint_cycles, 1) if args.smoke else args.max_joint_cycles
    )
    medium_candidates = (
        min(args.medium_candidates, 1) if args.smoke else args.medium_candidates
    )
    print(
        "问题二 Campo 基准："
        f"{base.ring_count} 环，结构化排除={base.excluded_ring_angles}",
        flush=True,
    )
    if not args.smoke:
        regression = _verify_q2_uniform_regression(
            base=base,
            q2_summary=q2_summary,
        )
        print(
            "统一规格正式回归通过："
            f"P={regression.annual_power_mw:.6f} MW，"
            f"q={regression.unit_area_power_kw_m2:.6f} kW/m²",
            flush=True,
        )
    search = optimize_multi_start(
        base=base,
        coarse_profile=search_coarse,
        medium_profile=search_medium,
        sobol_count=sobol_count,
        retained_count=retained_starts,
        target_power_mw=args.target_power,
        move_q_threshold=args.move_q,
        convergence_q_threshold=args.convergence_q,
        maximum_rounds=maximum_rounds,
        maximum_joint_cycles=maximum_joint_cycles,
        medium_candidate_limit=medium_candidates,
        seed=args.seed,
        progress=lambda message: print(message, flush=True),
    )
    selected_name, selected_design, selected, formal_candidates = _formal_candidates(
        base=base,
        search=search,
        target_power_mw=args.target_power,
        smoke=args.smoke,
    )
    dense_results: tuple[
        tuple[EvaluationProfile, Campo2DEvaluation],
        tuple[EvaluationProfile, Campo2DEvaluation],
    ] | None = None
    if not args.smoke:
        selected_name, selected_design, selected, dense_results = (
            _select_dense_feasible_candidate(
                base=base,
                formal_candidates=formal_candidates,
                target_power_mw=args.target_power,
            )
        )
    print(f"正式选择起点：{selected_name}", flush=True)
    _clean_output(args.output)
    written = write_primary_results(
        output_dir=args.output,
        base=base,
        design=selected_design,
        evaluation=selected,
        search=search,
        formal_candidates=formal_candidates,
        q2_baseline=q2_summary,
        six_group_baseline=six_summary,
        result3_template=args.result3_template,
        validation_evaluations=dense_results or (),
    )
    if args.smoke:
        written["dense"] = write_dense_validation(
            output_dir=args.output,
            evaluations=((smoke_profile(), selected),),
        )
    else:
        if dense_results is None:
            raise RuntimeError("正式运行缺少加密验证结果。")
        written["dense"] = write_dense_validation(
            output_dir=args.output,
            evaluations=(
                (formal_profile(), selected),
                *dense_results,
            ),
        )
    figure_paths = generate_figures(
        selected,
        q2_mirror_path=args.q2_mirrors,
        q2_monthly_path=args.q2_monthly,
        baseline_comparison_path=written["baseline"],
        output_dir=args.output,
    )
    for index, path in enumerate(figure_paths, start=1):
        written[f"figure_{index}"] = path

    print("\n第三问 Campo2D 结果", flush=True)
    print(f"镜子数：{selected.mirror_count}", flush=True)
    print(f"总面积：{selected.total_area_m2:.3f} m²", flush=True)
    print(f"年平均功率：{selected.annual_power_mw:.6f} MW", flush=True)
    print(
        f"单位面积输出：{selected.unit_area_power_kw_m2:.6f} kW/m²",
        flush=True,
    )
    for path in written.values():
        print(f"输出：{path}", flush=True)
    return 0


def main() -> None:
    raise SystemExit(run())
