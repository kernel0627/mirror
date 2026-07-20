"""独立第三问命令行：Campo 连续规格搜索、消融、复算和输出。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
    CampoMotherField,
    ContinuousDesign,
    ExpandedSpecifications,
    build_campo_mother_field,
)
from .prune import prune_symmetric_pairs
from .search import (
    SearchOutcome,
    optimize_continuous_design,
    refine_design_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
DEFAULT_Q2_COORDINATES = (
    PROJECT_ROOT / "outputs" / "q2" / "03_最终镜位坐标.csv"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "task" / "A" / "result3.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3_continuous"
LEGACY_Q3_OUTPUT = PROJECT_ROOT / "outputs" / "q3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="求解 CUMCM 2023 A 题第三问的 Campo 连续异构镜场"
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
    parser.add_argument(
        "--full-campo-prefix",
        action="store_true",
        help="使用修剪前 1471 面 Campo 前缀，而不是问题二正式 1469 面镜场。",
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
        "--max-cycles",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--azimuth-mode",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--skip-relaxed", action="store_true")
    parser.add_argument(
        "--prune-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prune-pairs-per-round",
        type=int,
        default=8,
    )
    parser.add_argument("--run-validation", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_power <= 0.0:
        raise SystemExit("--target-power 必须大于 0。")
    if args.max_cycles < 0:
        raise SystemExit("--max-cycles 不能小于 0。")
    if args.prune_rounds < 0:
        raise SystemExit("--prune-rounds 不能小于 0。")
    if args.prune_pairs_per_round < 1:
        raise SystemExit("--prune-pairs-per-round 必须大于等于 1。")


def _specifications_from_evaluation(
    source: HeterogeneousEvaluation,
    mother: CampoMotherField,
) -> ExpandedSpecifications:
    scales = source.widths / mother.base_width
    return ExpandedSpecifications(
        widths=source.widths,
        heights=source.heights,
        installation_heights=source.installation_heights,
        areas=source.widths * source.heights,
        scales=scales,
        size_shape=np.log(scales),
        area_normalizer=1.0,
    )


def _reevaluate(
    *,
    source: HeterogeneousEvaluation,
    profile: EvaluationProfile,
    mother: CampoMotherField,
    cache: EvaluationCache,
) -> HeterogeneousEvaluation:
    return evaluate_specifications(
        coordinates=source.coordinates,
        specifications=_specifications_from_evaluation(source, mother),
        ring_indices=source.ring_indices,
        zone_indices=source.zone_indices,
        zone_row_indices=source.zone_row_indices,
        normalized_rows=source.normalized_rows,
        azimuth_angles=source.azimuth_angles,
        azimuth_features=source.azimuth_features,
        nominal_ring_counts=source.nominal_ring_counts,
        actual_ring_counts=source.actual_ring_counts,
        original_indices=source.original_indices,
        field_config=field_config_from_mother(mother),
        profile=profile,
        safety_epsilon=mother.parameters.safety_epsilon,
        cache=cache,
    )


def _load_legacy_comparison(output: Path) -> dict | None:
    path = output / "07_最终方案摘要.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("layout") != "fixed-q2-campo-heterogeneous":
        legacy_path = output / "11_原六组对照结果.json"
        if legacy_path.exists():
            try:
                return json.loads(legacy_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None
    return {
        "model": "legacy-six-ring-groups",
        "role": "comparison-only",
        "mirror_count": payload.get("mirror_count"),
        "total_area_m2": payload.get("total_area_m2"),
        "group_design": payload.get("group_design"),
        "annual": payload.get("annual"),
        "geometry": payload.get("geometry"),
    }


def _formal_candidates(
    *,
    candidates: Sequence[
        tuple[str, ContinuousDesign, HeterogeneousEvaluation]
    ],
    pruned: HeterogeneousEvaluation,
    pruned_design: ContinuousDesign,
    mother: CampoMotherField,
    profile: EvaluationProfile,
    target_power_mw: float,
    cache: EvaluationCache,
) -> tuple[
    str,
    ContinuousDesign,
    HeterogeneousEvaluation,
    tuple[tuple[str, HeterogeneousEvaluation], ...],
]:
    formal: list[
        tuple[str, ContinuousDesign, HeterogeneousEvaluation]
    ] = []
    for name, design, evaluation in candidates:
        formal.append(
            (
                name,
                design,
                _reevaluate(
                    source=evaluation,
                    profile=profile,
                    mother=mother,
                    cache=cache,
                ),
            )
        )
    if not any(
        np.array_equal(pruned.original_indices, evaluation.original_indices)
        for _, _, evaluation in candidates
    ):
        formal.append(
            (
                "outer-boundary-pruned",
                pruned_design,
                _reevaluate(
                    source=pruned,
                    profile=profile,
                    mother=mother,
                    cache=cache,
                ),
            )
        )
    feasible = [
        item
        for item in formal
        if item[2].is_feasible(target_power_mw)
    ]
    if not feasible:
        powers = ", ".join(
            f"{name}={evaluation.annual_power_mw:.6f}"
            for name, _, evaluation in formal
        )
        raise RuntimeError(
            "正式精度下没有满足功率约束的候选："
            f"{powers} MW。"
        )
    best = max(
        feasible,
        key=lambda item: item[2].unit_area_power_kw_m2,
    )
    return (
        best[0],
        best[1],
        best[2],
        tuple((f"formal-{name}", evaluation) for name, _, evaluation in formal),
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    legacy_comparison = _load_legacy_comparison(LEGACY_Q3_OUTPUT)
    selected_coordinates = (
        None if args.full_campo_prefix else args.q2_coordinates
    )
    mother = build_campo_mother_field(
        args.q2_summary,
        selected_coordinates_path=selected_coordinates,
    )
    cache = EvaluationCache()
    if args.smoke:
        coarse = smoke_profile()
        reference = smoke_profile()
        final = smoke_profile()
        maximum_cycles = min(args.max_cycles, 1)
        prune_rounds = min(args.prune_rounds, 1)
        prune_pairs = min(args.prune_pairs_per_round, 2)
    else:
        coarse = coarse_profile()
        reference = medium_profile()
        final = formal_profile()
        maximum_cycles = args.max_cycles
        prune_rounds = args.prune_rounds
        prune_pairs = args.prune_pairs_per_round

    print(
        f"读取 Campo 镜场：{mother.mirror_count} 面，"
        f"区域镜数={mother.zone_counts}，"
        f"区域环数={mother.zone_ring_counts}"
    )
    radial: SearchOutcome = optimize_continuous_design(
        mother=mother,
        coarse_profile=coarse,
        reference_profile=reference,
        target_power_mw=args.target_power,
        include_azimuth=False,
        monotone=True,
        maximum_cycles_per_level=maximum_cycles,
        cache=cache,
        progress=print,
    )
    model_candidates: list[
        tuple[str, ContinuousDesign, HeterogeneousEvaluation]
    ] = [
        (
            "q2-uniform",
            radial.baseline_design,
            radial.baseline_evaluation,
        ),
        (
            "campo-monotone-radial",
            radial.best_design,
            radial.best_evaluation,
        ),
    ]
    stages = list(radial.stage_evaluations)
    current_design = radial.best_design
    current_evaluation = radial.best_evaluation
    current_name = "campo-monotone-radial"

    use_azimuth = (
        args.azimuth_mode == "on"
        or (
            args.azimuth_mode == "auto"
            and radial.diagnostics.azimuth_recommended
        )
    )
    if use_azimuth:
        current_design, current_evaluation, _ = refine_design_parameters(
            mother=mother,
            initial_design=current_design,
            coarse_profile=coarse,
            reference_profile=reference,
            parameters=("size_azimuth",),
            steps=(0.04, 0.02, 0.01),
            stage="azimuth-size",
            target_power_mw=args.target_power,
            monotone=True,
            maximum_cycles_per_level=maximum_cycles,
            cache=cache,
            progress=print,
        )
        current_design, current_evaluation, _ = refine_design_parameters(
            mother=mother,
            initial_design=current_design,
            coarse_profile=coarse,
            reference_profile=reference,
            parameters=("height_azimuth",),
            steps=(0.4, 0.2, 0.1),
            stage="azimuth-height",
            target_power_mw=args.target_power,
            monotone=True,
            maximum_cycles_per_level=maximum_cycles,
            cache=cache,
            progress=print,
        )
        current_name = "campo-monotone-azimuth"
        model_candidates.append(
            (current_name, current_design, current_evaluation)
        )
        stages.append((current_name, current_evaluation))

    if not args.skip_relaxed:
        relaxed_design, relaxed_evaluation, _ = refine_design_parameters(
            mother=mother,
            initial_design=current_design,
            coarse_profile=coarse,
            reference_profile=reference,
            parameters=(
                "size_zone1_slope",
                "size_zone2_slope",
            ),
            steps=(0.02, 0.01),
            stage="relaxed-size",
            target_power_mw=args.target_power,
            monotone=False,
            maximum_cycles_per_level=maximum_cycles,
            cache=cache,
            progress=print,
        )
        relaxed_design, relaxed_evaluation, _ = refine_design_parameters(
            mother=mother,
            initial_design=relaxed_design,
            coarse_profile=coarse,
            reference_profile=reference,
            parameters=(
                "height_zone1_slope",
                "height_zone2_slope",
            ),
            steps=(0.2, 0.1),
            stage="relaxed-height",
            target_power_mw=args.target_power,
            monotone=False,
            maximum_cycles_per_level=maximum_cycles,
            cache=cache,
            progress=print,
        )
        model_candidates.append(
            (
                "campo-relaxed",
                relaxed_design,
                relaxed_evaluation,
            )
        )
        stages.append(("campo-relaxed", relaxed_evaluation))
        if (
            relaxed_evaluation.is_feasible(args.target_power)
            and relaxed_evaluation.unit_area_power_kw_m2
            > current_evaluation.unit_area_power_kw_m2
        ):
            current_name = "campo-relaxed"
            current_design = relaxed_design
            current_evaluation = relaxed_evaluation

    pruned = current_evaluation
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
            f"外边界对称删镜接受 {len(pruning.steps)} 轮，"
            f"保留 {pruned.mirror_count} 面"
        )
        if pruning.steps:
            stages.append(("outer-boundary-pruned", pruned))

    (
        selected_name,
        selected_design,
        selected,
        formal_stages,
    ) = _formal_candidates(
        candidates=model_candidates,
        pruned=pruned,
        pruned_design=current_design,
        mother=mother,
        profile=final,
        target_power_mw=args.target_power,
        cache=cache,
    )
    stages.extend(formal_stages)
    stages.append(("formal-final", selected))

    written = write_question3_results(
        output_dir=args.output,
        mother=mother,
        design=selected_design,
        evaluation=selected,
        result3_template=args.result3_template,
        stages=stages,
        diagnostics=radial.diagnostics,
        model_name=selected_name,
        legacy_comparison=legacy_comparison,
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
    print(f"最终模型：{selected_name}")
    print(f"镜子数：{selected.mirror_count}")
    print(f"总镜面面积：{selected.total_area_m2:.3f} m²")
    print(f"年平均输出热功率：{selected.annual_power_mw:.6f} MW")
    print(
        "单位镜面面积年平均输出："
        f"{selected.unit_area_power_kw_m2:.6f} kW/m²"
    )
    print(
        "方位项诊断 RMSE 降幅："
        f"{100.0 * radial.diagnostics.relative_rmse_reduction:.3f}%"
    )
    for path in written.values():
        print(f"输出：{path}")
    return 0


def main() -> None:
    raise SystemExit(run())
