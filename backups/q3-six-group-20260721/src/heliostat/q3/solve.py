"""第三问命令行：异构分组搜索、删镜、正式复算和结果导出。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    coarse_profile,
    dense_profile,
    evaluate_specifications,
    field_config_from_mother,
    formal_profile,
    medium_profile,
    smoke_profile,
)
from .export import write_dense_validation, write_question3_results
from .model import (
    ExpandedSpecifications,
    GroupDesign,
    build_campo_mother_field,
)
from .prune import prune_symmetric_pairs
from .search import SearchOutcome, optimize_group_design


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
DEFAULT_TEMPLATE = PROJECT_ROOT / "task" / "A" / "result3.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="求解 CUMCM 2023 A 题第三问的分组异构定日镜场"
    )
    parser.add_argument(
        "--q2-summary",
        type=Path,
        default=DEFAULT_Q2_SUMMARY,
    )
    parser.add_argument(
        "--result3-template",
        type=Path,
        default=DEFAULT_TEMPLATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--target-power",
        type=float,
        default=42.0,
    )
    parser.add_argument(
        "--calibration-candidates",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prune-rounds",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prune-pairs-per-round",
        type=int,
        default=10,
    )
    parser.add_argument("--run-validation", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_power <= 0.0:
        raise SystemExit("--target-power 必须大于 0。")
    if args.calibration_candidates < 0:
        raise SystemExit("--calibration-candidates 不能小于 0。")
    if args.max_cycles < 0:
        raise SystemExit("--max-cycles 不能小于 0。")
    if args.prune_rounds < 0:
        raise SystemExit("--prune-rounds 不能小于 0。")
    if args.prune_pairs_per_round < 1:
        raise SystemExit("--prune-pairs-per-round 必须大于等于 1。")


def _reevaluate(
    *,
    source: HeterogeneousEvaluation,
    profile: EvaluationProfile,
    mother,
    cache: EvaluationCache,
) -> HeterogeneousEvaluation:
    specifications = ExpandedSpecifications(
        widths=source.widths,
        heights=source.heights,
        installation_heights=source.installation_heights,
        areas=source.widths * source.heights,
    )
    return evaluate_specifications(
        coordinates=source.coordinates,
        specifications=specifications,
        ring_indices=source.ring_indices,
        group_indices=source.group_indices,
        original_indices=source.original_indices,
        field_config=field_config_from_mother(mother),
        profile=profile,
        safety_epsilon=mother.parameters.safety_epsilon,
        cache=cache,
    )


def _formal_selection(
    *,
    outcome: SearchOutcome,
    pruned: HeterogeneousEvaluation,
    mother,
    profile: EvaluationProfile,
    target_power_mw: float,
    cache: EvaluationCache,
) -> HeterogeneousEvaluation:
    candidates = [
        _reevaluate(
            source=outcome.baseline_evaluation,
            profile=profile,
            mother=mother,
            cache=cache,
        ),
        _reevaluate(
            source=outcome.best_evaluation,
            profile=profile,
            mother=mother,
            cache=cache,
        ),
    ]
    if pruned.mirror_count != outcome.best_evaluation.mirror_count:
        candidates.append(
            _reevaluate(
                source=pruned,
                profile=profile,
                mother=mother,
                cache=cache,
            )
        )
    feasible = [
        candidate
        for candidate in candidates
        if candidate.is_feasible(target_power_mw)
    ]
    if not feasible:
        powers = ", ".join(
            f"{candidate.annual_power_mw:.6f}"
            for candidate in candidates
        )
        raise RuntimeError(
            "正式精度下没有满足功率约束的候选，"
            f"候选功率为：{powers} MW。"
        )
    return max(
        feasible,
        key=lambda evaluation: evaluation.unit_area_power_kw_m2,
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    mother = build_campo_mother_field(args.q2_summary)
    cache = EvaluationCache()
    if args.smoke:
        coarse = smoke_profile()
        reference = smoke_profile()
        final = smoke_profile()
        calibration_candidates = min(args.calibration_candidates, 1)
        maximum_cycles = min(args.max_cycles, 1)
        prune_rounds = min(args.prune_rounds, 1)
        prune_pairs = min(args.prune_pairs_per_round, 2)
    else:
        coarse = coarse_profile()
        reference = medium_profile()
        final = formal_profile()
        calibration_candidates = args.calibration_candidates
        maximum_cycles = args.max_cycles
        prune_rounds = args.prune_rounds
        prune_pairs = args.prune_pairs_per_round

    print(
        f"重建问题二完整 Campo 母场：{mother.mirror_count} 面，"
        f"组镜数={mother.group_counts}"
    )
    outcome = optimize_group_design(
        mother=mother,
        coarse_profile=coarse,
        reference_profile=reference,
        target_power_mw=args.target_power,
        calibration_candidate_count=calibration_candidates,
        maximum_cycles_per_level=maximum_cycles,
        cache=cache,
        progress=print,
    )
    print(
        "分组搜索完成："
        f"P={outcome.best_evaluation.annual_power_mw:.6f} MW，"
        f"q={outcome.best_evaluation.unit_area_power_kw_m2:.6f} kW/m²"
    )

    pruned = outcome.best_evaluation
    if prune_rounds and pruned.is_feasible(args.target_power):
        pruning = prune_symmetric_pairs(
            mother=mother,
            initial=pruned,
            profile=reference,
            target_power_mw=args.target_power,
            maximum_rounds=prune_rounds,
            maximum_pairs_per_round=prune_pairs,
            cache=cache,
        )
        pruned = pruning.best
        print(
            f"结构化删镜接受 {len(pruning.steps)} 轮，"
            f"保留 {pruned.mirror_count} 面"
        )

    selected = _formal_selection(
        outcome=outcome,
        pruned=pruned,
        mother=mother,
        profile=final,
        target_power_mw=args.target_power,
        cache=cache,
    )
    selected_is_uniform = (
        np.allclose(selected.widths, mother.base_width)
        and np.allclose(selected.heights, mother.base_height)
        and np.allclose(
            selected.installation_heights,
            mother.base_installation_height,
        )
    )
    selected_design = (
        outcome.baseline_design
        if selected_is_uniform
        else outcome.best_design
    )
    stages = list(outcome.stage_evaluations)
    if pruned.mirror_count != outcome.best_evaluation.mirror_count:
        stages.append(("structured-prune", pruned))
    stages.append(("formal-final", selected))

    calibration_payload = {
        **asdict(outcome.calibration),
        "paired_candidate_count": len(outcome.calibration_pairs),
        "note": "标定样本支持的经验误差带，不是数学严格置信界。",
    }
    written = write_question3_results(
        output_dir=args.output,
        mother=mother,
        design=selected_design,
        evaluation=selected,
        result3_template=args.result3_template,
        stages=stages,
        calibration=calibration_payload,
    )
    if args.run_validation and not args.smoke:
        dense_settings = dense_profile()
        dense = _reevaluate(
            source=selected,
            profile=dense_settings,
            mother=mother,
            cache=cache,
        )
        sensitivity_settings = replace(
            dense_settings,
            name="q3-dense-100m",
            solver=replace(
                dense_settings.solver,
                neighbor_radius_m=100.0,
            ),
        )
        sensitivity = _reevaluate(
            source=selected,
            profile=sensitivity_settings,
            mother=mother,
            cache=cache,
        )
        written["dense_validation"] = write_dense_validation(
            output_dir=args.output,
            evaluation=dense,
            profile=dense_settings,
            sensitivity_evaluations=(
                (sensitivity_settings, sensitivity),
            ),
        )

    print("\n第三问结果" if not args.smoke else "\n第三问烟雾测试结果")
    print(f"镜子数：{selected.mirror_count}")
    print(f"总镜面面积：{selected.total_area_m2:.3f} m²")
    print(f"年平均输出热功率：{selected.annual_power_mw:.6f} MW")
    print(
        "单位镜面面积年平均输出："
        f"{selected.unit_area_power_kw_m2:.6f} kW/m²"
    )
    for path in written.values():
        print(f"输出：{path}")
    return 0


def main() -> None:
    raise SystemExit(run())
