"""第二问双布局独立优化、统一复算与结果输出入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from ..config import SolverConfig
from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    better_evaluation,
    evaluate_coordinates,
    exploration_profile,
    final_profile,
    refinement_profile,
    scan_layout_extents,
)
from .export import write_high_precision_validation, write_question2_results
from .layout import generate_campo_layout, generate_partitioned_layout
from .layout import CampoParameters, PartitionedRingParameters
from .prune import prune_outer_symmetric_pairs
from .search import (
    CAMPO_STEP_LEVELS,
    PARTITIONED_STEP_LEVELS,
    refine_campo,
    refine_partitioned,
    optimize_campo,
    optimize_partitioned,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q2"
DEFAULT_TEMPLATE = PROJECT_ROOT / "task" / "A" / "result2.xlsx"


def _smoke_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="smoke",
        solver=SolverConfig(
            shadow_grid_size=3,
            truncation_rays=8,
            neighbor_radius_m=60.0,
            truncation_chunk_size=64,
            sobol_seed=2023,
        ),
        months=(6,),
        solar_times=(12.0,),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立优化分区圆环和改进 Campo 两种问题二镜场"
    )
    parser.add_argument(
        "--layout",
        choices=("both", "partitioned", "campo"),
        default="both",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--initial-samples", type=int, default=16)
    parser.add_argument("--retained-starts", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--coarse-stride", type=int, default=4)
    parser.add_argument("--extent-window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--resume-comparison",
        type=Path,
        default=None,
        help="从上一阶段的 02_双布局比较.json 参数继续局部搜索",
    )
    parser.add_argument(
        "--search-profile",
        choices=("exploration", "refinement"),
        default="exploration",
        help="非烟雾搜索使用的数值离散精度",
    )
    parser.add_argument(
        "--step-level-count",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="本阶段连续使用几档步长",
    )
    parser.add_argument(
        "--step-level-start",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="本阶段从第几档步长开始（1 为粗、3 为细）",
    )
    parser.add_argument(
        "--prune-rounds",
        type=int,
        default=10,
        help="胜出布局结构化删镜的最大轮数；0 表示跳过",
    )
    parser.add_argument(
        "--prune-pairs-per-round",
        type=int,
        default=None,
        help="每轮最多复算的外层对称镜位对；默认全部",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="仅用 6 月正午、3×3 阴影网格和 8 条截断光线验证流程",
    )
    parser.add_argument(
        "--skip-x-check",
        action="store_true",
        help="跳过塔东西坐标 {-10,-5,0,5,10} m 的少量复核",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="不生成四张正式结果图",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="读取输出目录中的正式结果并重新生成四张图",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="额外运行 20×20 阴影网格、512 条截断光线的加密复算",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "initial_samples",
        "retained_starts",
        "coarse_stride",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} 必须大于等于 1。")
    if args.extent_window < 0:
        raise SystemExit("--extent-window 不能小于 0。")
    if args.max_cycles < 0:
        raise SystemExit("--max-cycles 不能小于 0。")
    if args.step_level_start + args.step_level_count - 1 > 3:
        raise SystemExit("--step-level-start 与 --step-level-count 超出三档步长。")
    if args.prune_rounds < 0:
        raise SystemExit("--prune-rounds 不能小于 0。")
    if args.prune_pairs_per_round is not None and args.prune_pairs_per_round < 1:
        raise SystemExit("--prune-pairs-per-round 必须大于等于 1。")


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.figures_only:
        from .plot import build_question2_figures_from_output

        for path in build_question2_figures_from_output(args.output):
            print(f"输出：{path}")
        return 0
    if args.smoke:
        search_profile = _smoke_profile()
    elif args.search_profile == "refinement":
        search_profile = refinement_profile()
    else:
        search_profile = exploration_profile()
    verification_profile = _smoke_profile() if args.smoke else final_profile()
    cache = EvaluationCache()
    optimized: dict[str, object] = {}
    step_start = args.step_level_start - 1
    step_stop = step_start + args.step_level_count
    partitioned_steps = PARTITIONED_STEP_LEVELS[step_start:step_stop]
    campo_steps = CAMPO_STEP_LEVELS[step_start:step_stop]
    resumed = None
    if args.resume_comparison is not None:
        if not args.resume_comparison.exists():
            raise SystemExit(f"找不到恢复文件：{args.resume_comparison}")
        resumed = json.loads(args.resume_comparison.read_text(encoding="utf-8"))

    if args.layout in ("both", "partitioned"):
        print("开始独立优化方案 A：分区交错同心圆")
        if resumed is not None:
            optimized["partitioned"] = refine_partitioned(
                PartitionedRingParameters(**resumed["partitioned"]["parameters"]),
                profile=search_profile,
                step_levels=partitioned_steps,
                maximum_cycles_per_level=args.max_cycles,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
                progress=print,
            )
        else:
            optimized["partitioned"] = optimize_partitioned(
                profile=search_profile,
                initial_sample_count=args.initial_samples,
                retained_starts=args.retained_starts,
                seed=args.seed,
                step_levels=partitioned_steps,
                maximum_cycles_per_level=args.max_cycles,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
                progress=print,
            )
    if args.layout in ("both", "campo"):
        print("开始独立优化方案 B：改进 Campo 径向交错")
        if resumed is not None:
            optimized["campo"] = refine_campo(
                CampoParameters(**resumed["campo"]["parameters"]),
                profile=search_profile,
                step_levels=campo_steps,
                maximum_cycles_per_level=args.max_cycles,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
                progress=print,
            )
        else:
            optimized["campo"] = optimize_campo(
                profile=search_profile,
                initial_sample_count=args.initial_samples,
                retained_starts=args.retained_starts,
                seed=args.seed + 1,
                step_levels=campo_steps,
                maximum_cycles_per_level=args.max_cycles,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
                progress=print,
            )

    verified: dict[str, tuple[object, object, object, object]] = {}
    for kind, result in optimized.items():
        parameters = result.best.parameters
        x_check_scan = None
        if not args.smoke and not args.skip_x_check:
            center_layout = (
                generate_partitioned_layout(parameters)
                if kind == "partitioned"
                else generate_campo_layout(parameters)
            )
            center_scan = scan_layout_extents(
                center_layout,
                parameters,
                verification_profile,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
            )
            selected_parameters = parameters
            selected_scan = center_scan
            for tower_x in (-10.0, -5.0, 5.0, 10.0):
                candidate_parameters = replace(
                    parameters,
                    tower_x=tower_x,
                )
                candidate_layout = (
                    generate_partitioned_layout(candidate_parameters)
                    if kind == "partitioned"
                    else generate_campo_layout(candidate_parameters)
                )
                candidate_scan = scan_layout_extents(
                    candidate_layout,
                    candidate_parameters,
                    verification_profile,
                    coarse_stride=args.coarse_stride,
                    window=args.extent_window,
                    cache=cache,
                )
                selected = better_evaluation(
                    selected_scan.best,
                    candidate_scan.best,
                )
                selected_feasible = selected_scan.best.is_feasible()
                candidate_feasible = candidate_scan.best.is_feasible()
                if candidate_feasible != selected_feasible:
                    stable_gain = candidate_feasible
                elif candidate_feasible:
                    stable_gain = (
                        candidate_scan.best.unit_area_power_kw_m2
                        - selected_scan.best.unit_area_power_kw_m2
                        > 1e-4
                    )
                else:
                    stable_gain = (
                        candidate_scan.best.annual_power_mw
                        - selected_scan.best.annual_power_mw
                        > 1e-3
                    )
                if selected is candidate_scan.best and stable_gain:
                    selected_parameters = candidate_parameters
                    selected_scan = candidate_scan
            parameters = selected_parameters
            x_check_scan = selected_scan
        layout = (
            generate_partitioned_layout(parameters)
            if kind == "partitioned"
            else generate_campo_layout(parameters)
        )
        precision_label = "烟雾测试精度" if args.smoke else "问题一最终精度"
        print(f"使用统一{precision_label}复算 {kind}")
        if x_check_scan is None:
            scan = scan_layout_extents(
                layout,
                parameters,
                verification_profile,
                coarse_stride=args.coarse_stride,
                window=args.extent_window,
                cache=cache,
            )
        else:
            scan = x_check_scan
        verified[kind] = (parameters, result, layout, scan)

    verified_values = list(verified.items())
    winner_kind, winner_bundle = verified_values[0]
    for kind, bundle in verified_values[1:]:
        if (
            better_evaluation(
                winner_bundle[3].best,
                bundle[3].best,
            )
            is bundle[3].best
        ):
            winner_kind, winner_bundle = kind, bundle

    winner_parameters, _, winner_layout, winner_scan = winner_bundle
    winner_evaluation = winner_scan.best

    if args.prune_rounds and abs(winner_parameters.tower_x) <= 1e-9:
        print("对胜出布局执行外层东西对称镜位修剪")
        prune = prune_outer_symmetric_pairs(
            layout=winner_layout,
            parameters=winner_parameters,
            initial=winner_evaluation,
            profile=verification_profile,
            maximum_rounds=args.prune_rounds,
            maximum_pairs_per_round=args.prune_pairs_per_round,
            cache=cache,
        )
        winner_evaluation = prune.best
    elif args.prune_rounds:
        print("塔东西坐标不为 0，跳过要求南北轴对称的外层镜位修剪")

    args.output.mkdir(parents=True, exist_ok=True)
    comparison = {
        kind: {
            "parameters": asdict(bundle[0]),
            "ring_count": bundle[3].best.ring_count,
            "mirror_count": bundle[3].best.mirror_count,
            "total_area_m2": bundle[3].best.total_area_m2,
            "annual_power_mw": bundle[3].best.annual_power_mw,
            "unit_area_power_kw_m2": (bundle[3].best.unit_area_power_kw_m2),
        }
        for kind, bundle in verified.items()
    }
    comparison[winner_kind].update(
        {
            "ring_count": winner_evaluation.ring_count,
            "mirror_count": winner_evaluation.mirror_count,
            "total_area_m2": winner_evaluation.total_area_m2,
            "annual_power_mw": winner_evaluation.annual_power_mw,
            "unit_area_power_kw_m2": (winner_evaluation.unit_area_power_kw_m2),
        }
    )
    comparison["winner"] = winner_kind
    comparison_path = args.output / "02_双布局比较.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written = write_question2_results(
        output_dir=args.output,
        layout_name=winner_kind,
        parameters=winner_parameters,
        evaluation=winner_evaluation,
        result2_template=args.template,
        comparison=comparison,
    )
    if args.run_validation and not args.smoke:
        dense_profile = EvaluationProfile(
            name="dense-validation",
            solver=SolverConfig(
                shadow_grid_size=20,
                truncation_rays=512,
                neighbor_radius_m=80.0,
                truncation_chunk_size=128,
                sobol_seed=2023,
            ),
        )
        dense_evaluation = evaluate_coordinates(
            layout_kind=winner_kind,
            ring_count=winner_evaluation.ring_count,
            coordinates=winner_evaluation.coordinates,
            parameters=winner_parameters,
            profile=dense_profile,
        )
        written["dense_validation"] = write_high_precision_validation(
            output_dir=args.output,
            evaluation=dense_evaluation,
            profile=dense_profile,
        )
    if not args.skip_figures and len(verified) == 2:
        from .plot import build_question2_figures

        figure_evaluations = {kind: bundle[3].best for kind, bundle in verified.items()}
        figure_evaluations[winner_kind] = winner_evaluation
        figure_parameters = {kind: bundle[0] for kind, bundle in verified.items()}
        for path in build_question2_figures(
            output_dir=args.output,
            comparison=comparison,
            parameters=figure_parameters,
            evaluations=figure_evaluations,
        ):
            written[path.stem] = path
    elif not args.skip_figures:
        print("仅优化一种布局，跳过双布局对比图。")

    print(
        "\n第二问烟雾测试结果（不可作为正式年平均结论）"
        if args.smoke
        else "\n第二问结果"
    )
    print(f"胜出布局：{winner_kind}")
    print(f"镜子数：{winner_evaluation.mirror_count}")
    print(f"总镜面面积：{winner_evaluation.total_area_m2:.3f} m²")
    target_power_mw = 42.0
    print(f"年平均输出热功率约束下限：{target_power_mw:.6f} MW")
    print(f"最终年平均输出热功率：{winner_evaluation.annual_power_mw:.6f} MW")
    print(
        "相对约束下限的功率余量："
        f"{winner_evaluation.annual_power_mw - target_power_mw:.6f} MW"
    )
    print(
        f"单位面积年平均输出热功率：{winner_evaluation.unit_area_power_kw_m2:.6f} kW/m²"
    )
    print(f"双布局比较：{comparison_path}")
    for path in written.values():
        print(f"输出：{path}")
    return 0


def main() -> None:
    raise SystemExit(run())
