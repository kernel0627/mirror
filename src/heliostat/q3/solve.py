"""六区阶梯参数微调的完整分阶段入口。"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluate import (
    EvaluationCache,
    RefineEvaluation,
    dense_profile,
    evaluate_design,
    formal_profile,
    medium_profile,
    metrics,
    smoke_profile,
)
from .export import write_results
from .model import RefineBaseline, RefineDesign, load_baseline
from .plot import generate_figures
from .search import coordinate_search
from .sensitivity import (
    active_from_formal,
    select_formal_directions,
    specification_perturbations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
DEFAULT_SIX_GROUP_SUMMARY = (
    PROJECT_ROOT / "src" / "heliostat" / "q3" / "six_group_baseline.json"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "task" / "A" / "result3.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三问六区阶梯参数敏感性与局部微调")
    parser.add_argument("--q2-summary", type=Path, default=DEFAULT_Q2_SUMMARY)
    parser.add_argument(
        "--six-group-summary",
        type=Path,
        default=DEFAULT_SIX_GROUP_SUMMARY,
    )
    parser.add_argument("--result3-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--target-power", type=float, default=42.0)
    parser.add_argument("--medium-limit", type=int, default=150)
    parser.add_argument("--formal-limit", type=int, default=12)
    parser.add_argument("--max-sweeps", type=int, default=2)
    parser.add_argument("--move-q", type=float, default=1e-8)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_power <= 0.0:
        raise SystemExit("--target-power 必须大于 0。")
    if args.medium_limit < 69 or args.medium_limit > 150:
        raise SystemExit("--medium-limit 必须位于 69 到 150。")
    if args.formal_limit < 12 or args.formal_limit > 12:
        raise SystemExit("本方案严格使用 --formal-limit 12。")
    if args.max_sweeps < 0 or args.max_sweeps > 2:
        raise SystemExit("--max-sweeps 必须位于 0 到 2。")
    if args.move_q < 0.0:
        raise SystemExit("--move-q 不能小于 0。")


def _rank_key(evaluation: RefineEvaluation, target_power_mw: float) -> tuple[int, float]:
    feasible = evaluation.is_feasible(target_power_mw)
    return (
        int(feasible),
        evaluation.unit_area_power_kw_m2 if feasible else evaluation.annual_power_mw,
    )


def _better(
    candidate: RefineEvaluation,
    reference: RefineEvaluation,
    *,
    target_power_mw: float,
    threshold: float,
) -> bool:
    candidate_feasible = candidate.is_feasible(target_power_mw)
    reference_feasible = reference.is_feasible(target_power_mw)
    if candidate_feasible != reference_feasible:
        return candidate_feasible
    if candidate_feasible:
        return candidate.unit_area_power_kw_m2 > reference.unit_area_power_kw_m2 + threshold
    return candidate.annual_power_mw > reference.annual_power_mw + 1e-6


def _regression_payload(
    baseline: RefineBaseline,
    evaluation: RefineEvaluation,
) -> dict[str, object]:
    parameter_errors = {
        "width_max_abs_m": float(
            np.max(
                np.abs(
                    evaluation.specifications.widths
                    - np.asarray(baseline.design.widths)[evaluation.field.group_indices]
                )
            )
        ),
        "height_max_abs_m": float(
            np.max(
                np.abs(
                    evaluation.specifications.heights
                    - np.asarray(baseline.design.mirror_heights)[evaluation.field.group_indices]
                )
            )
        ),
        "installation_height_max_abs_m": float(
            np.max(
                np.abs(
                    evaluation.specifications.installation_heights
                    - np.asarray(baseline.design.installation_heights)[
                        evaluation.field.group_indices
                    ]
                )
            )
        ),
    }
    coordinate_error = float(
        np.max(np.abs(evaluation.field.coordinates - baseline.mother.coordinates))
    )
    errors = {
        "mirror_count": evaluation.mirror_count - baseline.expected_mirror_count,
        "coordinate_max_abs_m": coordinate_error,
        "total_area_m2": evaluation.total_area_m2 - baseline.expected_total_area_m2,
        "annual_power_mw": evaluation.annual_power_mw - baseline.expected_power_mw,
        "unit_area_power_kw_m2": (
            evaluation.unit_area_power_kw_m2 - baseline.expected_q_kw_m2
        ),
        **parameter_errors,
    }
    tolerances = {
        "mirror_count": 0,
        "coordinate_max_abs_m": 1e-12,
        "total_area_m2": 1e-6,
        "annual_power_mw": 1e-6,
        "unit_area_power_kw_m2": 1e-9,
        "width_max_abs_m": 0.0,
        "height_max_abs_m": 0.0,
        "installation_height_max_abs_m": 0.0,
    }
    passed = all(abs(float(errors[key])) <= tolerance for key, tolerance in tolerances.items())
    return {
        "passed": passed,
        "expected": {
            "mirror_count": baseline.expected_mirror_count,
            "total_area_m2": baseline.expected_total_area_m2,
            "annual_power_mw": baseline.expected_power_mw,
            "unit_area_power_kw_m2": baseline.expected_q_kw_m2,
        },
        "actual": metrics(evaluation),
        "errors": errors,
        "tolerances": tolerances,
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    baseline = load_baseline(
        q2_summary_path=args.q2_summary,
        six_group_summary_path=args.six_group_summary,
    )
    search_profile = smoke_profile() if args.smoke else medium_profile()
    verification_profile = smoke_profile() if args.smoke else formal_profile()
    search_cache = EvaluationCache()
    formal_cache = EvaluationCache()
    medium_count = 0
    formal_count = 0

    def try_evaluate(
        design: RefineDesign,
        *,
        profile_kind: str,
        count_candidate: bool = True,
    ) -> tuple[RefineEvaluation | None, str | None]:
        nonlocal medium_count, formal_count
        if profile_kind == "medium":
            if count_candidate and medium_count >= args.medium_limit:
                return None, "达到中精度候选上限"
            profile = search_profile
            cache = search_cache
        elif profile_kind == "formal":
            if count_candidate and formal_count >= args.formal_limit:
                return None, "达到正式候选上限"
            profile = verification_profile
            cache = formal_cache
        else:
            raise ValueError(profile_kind)
        try:
            evaluation = evaluate_design(
                baseline=baseline,
                design=design,
                profile=profile,
                cache=cache,
            )
        except ValueError as exc:
            return None, str(exc)
        if count_candidate:
            if profile_kind == "medium":
                medium_count += 1
            else:
                formal_count += 1
        return evaluation, None

    print("阶段 0/4：六组正式初值回归", flush=True)
    baseline_formal, reason = try_evaluate(
        baseline.design,
        profile_kind="formal",
        count_candidate=False,
    )
    if baseline_formal is None:
        raise RuntimeError(f"六组回归无法评价：{reason}")
    regression = _regression_payload(baseline, baseline_formal)
    if not args.smoke and not regression["passed"]:
        raise RuntimeError(f"六组正式回归失败：{regression['errors']}")
    print(
        f"回归通过：P={baseline_formal.annual_power_mw:.9f} MW，"
        f"q={baseline_formal.unit_area_power_kw_m2:.9f} kW/m²",
        flush=True,
    )
    baseline_medium, reason = try_evaluate(
        baseline.design,
        profile_kind="medium",
        count_candidate=False,
    )
    if baseline_medium is None:
        raise RuntimeError(f"六组中精度基准无法评价：{reason}")

    formal_rows: list[dict[str, object]] = []
    print("阶段 1/4：塔位模式 A/B 独立扫描", flush=True)
    tower_internal: list[dict[str, object]] = []
    tower_rows: list[dict[str, object]] = []
    for mode in ("A", "B"):
        mode_records: list[dict[str, object]] = []
        for delta in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            design = replace(
                baseline.design,
                tower_mode=mode,
                tower_y=baseline.design.tower_y + delta,
            )
            evaluation, reject = try_evaluate(design, profile_kind="medium")
            row: dict[str, object] = {
                "tower_mode": mode,
                "tower_x": 0.0,
                "tower_y": design.tower_y,
                "delta_y_m": delta,
                "legal": evaluation is not None,
                "reject_reason": reject or "",
                "selected_for_formal": False,
            }
            if evaluation is not None:
                row.update(metrics(evaluation, target_power_mw=args.target_power))
                row["delta_q_from_six_medium"] = (
                    evaluation.unit_area_power_kw_m2
                    - baseline_medium.unit_area_power_kw_m2
                )
                mode_records.append({"row": row, "design": design, "medium": evaluation})
            tower_rows.append(row)
        ranked = sorted(
            mode_records,
            key=lambda item: _rank_key(item["medium"], args.target_power),
            reverse=True,
        )
        for record in ranked[:2]:
            evaluation, reject = try_evaluate(record["design"], profile_kind="formal")
            record["row"]["selected_for_formal"] = True
            record["row"]["formal_reject_reason"] = reject or ""
            if evaluation is not None:
                record["formal"] = evaluation
                record["row"]["formal_power_mw"] = evaluation.annual_power_mw
                record["row"]["formal_q_kw_m2"] = evaluation.unit_area_power_kw_m2
                formal_rows.append(
                    {
                        "stage": "tower_scan",
                        "candidate": f"mode-{mode}-dy-{record['row']['delta_y_m']:+g}",
                        **metrics(evaluation, target_power_mw=args.target_power),
                        "delta_q_from_six": (
                            evaluation.unit_area_power_kw_m2
                            - baseline_formal.unit_area_power_kw_m2
                        ),
                    }
                )
        tower_internal.extend(mode_records)

    best_by_mode: dict[str, dict[str, object]] = {}
    for mode in ("A", "B"):
        records = [record for record in tower_internal if record["design"].tower_mode == mode and "formal" in record]
        if records:
            best_by_mode[mode] = max(
                records,
                key=lambda item: _rank_key(item["formal"], args.target_power),
            )
    improving_modes = {
        mode: record
        for mode, record in best_by_mode.items()
        if _better(
            record["formal"],
            baseline_formal,
            target_power_mw=args.target_power,
            threshold=args.move_q,
        )
    }
    if not improving_modes:
        current_design = baseline.design
        current_formal = baseline_formal
        tower_active = False
        tower_decision = "两种语义均无正式改善，固定原塔位并采用模式 A"
    elif "A" in improving_modes and "B" in improving_modes and abs(
        improving_modes["A"]["formal"].unit_area_power_kw_m2
        - improving_modes["B"]["formal"].unit_area_power_kw_m2
    ) <= 1e-5:
        chosen = improving_modes["A"]
        current_design = chosen["design"]
        current_formal = chosen["formal"]
        tower_active = not math.isclose(current_design.tower_y, baseline.design.tower_y)
        tower_decision = "两种语义接近，按文档优先采用模式 A"
    else:
        chosen = max(
            improving_modes.values(),
            key=lambda item: _rank_key(item["formal"], args.target_power),
        )
        current_design = chosen["design"]
        current_formal = chosen["formal"]
        tower_active = not math.isclose(current_design.tower_y, baseline.design.tower_y)
        tower_decision = f"正式精度选择模式 {current_design.tower_mode}"
    print(
        f"塔位语义：{tower_decision}；y={current_design.tower_y:.6f} m",
        flush=True,
    )

    print("阶段 2/4：D1、g 一维粗扫及 3×3 局部组合", flush=True)
    geometry_origin = current_design
    geometry_origin_formal = current_formal
    geometry_internal: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []

    def add_geometry(label: str, design: RefineDesign) -> None:
        evaluation, reject = try_evaluate(design, profile_kind="medium")
        row: dict[str, object] = {
            "scan": label,
            "tower_mode": design.tower_mode,
            "tower_y": design.tower_y,
            "initial_spacing": design.initial_spacing,
            "spacing_growth": design.spacing_growth,
            "delta_D1_from_six": design.initial_spacing - baseline.design.initial_spacing,
            "delta_g_from_six": design.spacing_growth - baseline.design.spacing_growth,
            "legal": evaluation is not None,
            "reject_reason": reject or "",
            "selected_for_formal": False,
        }
        if evaluation is not None:
            row.update(metrics(evaluation, target_power_mw=args.target_power))
            row["delta_power_from_six_medium"] = (
                evaluation.annual_power_mw - baseline_medium.annual_power_mw
            )
            row["delta_q_from_six_medium"] = (
                evaluation.unit_area_power_kw_m2
                - baseline_medium.unit_area_power_kw_m2
            )
            geometry_internal.append({"row": row, "design": design, "medium": evaluation})
        geometry_rows.append(row)

    for delta in (-0.20, -0.10, 0.0, 0.10, 0.20):
        add_geometry(
            "D1-one-dimensional",
            replace(geometry_origin, initial_spacing=baseline.design.initial_spacing + delta),
        )
    for delta in (-0.02, -0.01, 0.0, 0.01, 0.02):
        add_geometry(
            "g-one-dimensional",
            replace(geometry_origin, spacing_growth=baseline.design.spacing_growth + delta),
        )
    d_records = [record for record in geometry_internal if record["row"]["scan"] == "D1-one-dimensional"]
    g_records = [record for record in geometry_internal if record["row"]["scan"] == "g-one-dimensional"]
    best_d = max(d_records, key=lambda item: _rank_key(item["medium"], args.target_power))["design"].initial_spacing
    best_g = max(g_records, key=lambda item: _rank_key(item["medium"], args.target_power))["design"].spacing_growth
    for delta_d in (-0.10, 0.0, 0.10):
        for delta_g in (-0.01, 0.0, 0.01):
            add_geometry(
                "D1-g-3x3",
                replace(
                    geometry_origin,
                    initial_spacing=best_d + delta_d,
                    spacing_growth=best_g + delta_g,
                ),
            )
    geometry_ranked = sorted(
        geometry_internal,
        key=lambda item: _rank_key(item["medium"], args.target_power),
        reverse=True,
    )
    geometry_best = geometry_ranked[0]
    geometry_formal, reject = try_evaluate(geometry_best["design"], profile_kind="formal")
    geometry_best["row"]["selected_for_formal"] = True
    geometry_best["row"]["formal_reject_reason"] = reject or ""
    if geometry_formal is not None:
        geometry_best["formal"] = geometry_formal
        geometry_best["row"]["formal_power_mw"] = geometry_formal.annual_power_mw
        geometry_best["row"]["formal_q_kw_m2"] = geometry_formal.unit_area_power_kw_m2
        formal_rows.append(
            {
                "stage": "campo_geometry",
                "candidate": "best-medium-geometry",
                **metrics(geometry_formal, target_power_mw=args.target_power),
                "delta_q_from_six": (
                    geometry_formal.unit_area_power_kw_m2
                    - baseline_formal.unit_area_power_kw_m2
                ),
            }
        )
        if _better(
            geometry_formal,
            geometry_origin_formal,
            target_power_mw=args.target_power,
            threshold=args.move_q,
        ):
            current_design = geometry_best["design"]
            current_formal = geometry_formal
    geometry_active = tuple(
        parameter
        for parameter in ("initial_spacing", "spacing_growth")
        if not math.isclose(
            current_design.parameter(parameter), geometry_origin.parameter(parameter)
        )
    )
    print(
        f"几何固定点：D1={current_design.initial_spacing:.6f} m，"
        f"g={current_design.spacing_growth:.6f} m/环",
        flush=True,
    )

    print("阶段 3/4：18 个六区规格变量正负敏感性", flush=True)
    sensitivity_reference, reason = try_evaluate(
        current_design,
        profile_kind="medium",
        count_candidate=False,
    )
    if sensitivity_reference is None:
        raise RuntimeError(f"敏感性中精度基准无法评价：{reason}")
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_designs: dict[tuple[str, str], RefineDesign] = {}
    for perturbation in specification_perturbations(current_design):
        evaluation, reject = try_evaluate(perturbation.design, profile_kind="medium")
        row: dict[str, object] = {
            "parameter": perturbation.parameter,
            "group_id": perturbation.group_id,
            "old_value": perturbation.old_value,
            "new_value": perturbation.new_value,
            "direction": perturbation.direction,
            "legal": evaluation is not None,
            "medium_power": None,
            "medium_q": None,
            "delta_power": None,
            "delta_q": None,
            "formal_power": None,
            "formal_q": None,
            "active": False,
            "reject_reason": reject or "",
        }
        if evaluation is not None:
            row.update(
                {
                    "medium_power": evaluation.annual_power_mw,
                    "medium_q": evaluation.unit_area_power_kw_m2,
                    "delta_power": (
                        evaluation.annual_power_mw
                        - baseline_medium.annual_power_mw
                    ),
                    "delta_q": (
                        evaluation.unit_area_power_kw_m2
                        - baseline_medium.unit_area_power_kw_m2
                    ),
                    "delta_power_from_geometry": (
                        evaluation.annual_power_mw
                        - sensitivity_reference.annual_power_mw
                    ),
                    "delta_q_from_geometry": (
                        evaluation.unit_area_power_kw_m2
                        - sensitivity_reference.unit_area_power_kw_m2
                    ),
                }
            )
            sensitivity_designs[(perturbation.parameter, perturbation.direction)] = perturbation.design
        sensitivity_rows.append(row)
    selected_directions = select_formal_directions(sensitivity_rows, limit=6)
    for row in selected_directions:
        design = sensitivity_designs[(str(row["parameter"]), str(row["direction"]))]
        evaluation, reject = try_evaluate(design, profile_kind="formal")
        if evaluation is None:
            row["reject_reason"] = reject or "正式复算失败"
            continue
        row["formal_power"] = evaluation.annual_power_mw
        row["formal_q"] = evaluation.unit_area_power_kw_m2
        row["active"] = (
            evaluation.is_feasible(args.target_power)
            and evaluation.unit_area_power_kw_m2
            > current_formal.unit_area_power_kw_m2 + args.move_q
        )
        formal_rows.append(
            {
                "stage": "specification_sensitivity",
                "candidate": f"{row['parameter']}{row['direction']}",
                **metrics(evaluation, target_power_mw=args.target_power),
                "delta_q_from_six": (
                    evaluation.unit_area_power_kw_m2
                    - baseline_formal.unit_area_power_kw_m2
                ),
            }
        )
    specification_active = active_from_formal(
        sensitivity_rows,
        reference_q=current_formal.unit_area_power_kw_m2,
        target_power_mw=args.target_power,
        threshold=args.move_q,
    )
    active_variables = (
        *(("tower_y",) if tower_active else ()),
        *geometry_active,
        *specification_active,
    )
    print(f"正式确认的活跃变量：{active_variables or '无'}", flush=True)

    print("阶段 4/4：活跃变量两轮分块回扫及统一验收", flush=True)
    local_initial = sensitivity_reference

    def local_evaluator(design: RefineDesign) -> RefineEvaluation | None:
        evaluation, reject_reason = try_evaluate(design, profile_kind="medium")
        if evaluation is None and reject_reason == "达到中精度候选上限":
            return None
        return evaluation

    search = coordinate_search(
        initial_design=current_design,
        initial_evaluation=local_initial,
        active_variables=active_variables,
        evaluator=local_evaluator,
        baseline_q_kw_m2=baseline_medium.unit_area_power_kw_m2,
        maximum_sweeps=args.max_sweeps,
        target_power_mw=args.target_power,
        move_q_threshold=args.move_q,
    )
    attempted_formal, reason = try_evaluate(
        search.best_design,
        profile_kind="formal",
    )
    if attempted_formal is None:
        raise RuntimeError(f"最终候选正式复算失败：{reason}")
    formal_rows.append(
        {
            "stage": "final_acceptance",
            "candidate": "local-search-best",
            **metrics(attempted_formal, target_power_mw=args.target_power),
            "delta_q_from_six": (
                attempted_formal.unit_area_power_kw_m2
                - baseline_formal.unit_area_power_kw_m2
            ),
        }
    )

    dense_payload: dict[str, object] = {
        "status": "not-run-formal-candidate-failed",
        "baseline": {},
        "candidate": {},
    }
    formal_pass = _better(
        attempted_formal,
        baseline_formal,
        target_power_mw=args.target_power,
        threshold=0.0,
    )
    dense_pass = False
    if args.smoke:
        dense_payload["status"] = "smoke-skipped"
    elif formal_pass:
        dense_payload["status"] = "completed"
        dense_cache = EvaluationCache()
        dense_pass = True
        for radius in (80.0, 100.0):
            profile = dense_profile(neighbor_radius_m=radius)
            baseline_dense = evaluate_design(
                baseline=baseline,
                design=baseline.design,
                profile=profile,
                cache=dense_cache,
            )
            candidate_dense = evaluate_design(
                baseline=baseline,
                design=search.best_design,
                profile=profile,
                cache=dense_cache,
            )
            key = f"{int(radius)}"
            dense_payload["baseline"][key] = metrics(
                baseline_dense, target_power_mw=args.target_power
            )
            dense_payload["candidate"][key] = metrics(
                candidate_dense, target_power_mw=args.target_power
            )
            dense_pass = dense_pass and candidate_dense.is_feasible(args.target_power) and (
                candidate_dense.unit_area_power_kw_m2
                > baseline_dense.unit_area_power_kw_m2
            )
        dense_payload["passed"] = dense_pass

    accepted = formal_pass and (args.smoke or dense_pass)
    if accepted:
        selected_design = search.best_design
        selected_formal = attempted_formal
        decision = "微调方案通过统一正式与加密验收" if not args.smoke else "smoke 链路通过"
    else:
        selected_design = baseline.design
        selected_formal = baseline_formal
        decision = "微调候选未通过统一验收，保留原六组正式方案"

    active_payload = {
        "tower_mode_decision": tower_decision,
        "selected_tower_mode": current_design.tower_mode,
        "active_variables": list(active_variables),
        "tower_active": tower_active,
        "geometry_active": list(geometry_active),
        "specification_active": list(specification_active),
        "medium_candidate_count": medium_count,
        "medium_candidate_limit": args.medium_limit,
        "formal_candidate_count": formal_count,
        "formal_candidate_limit": args.formal_limit,
        "maximum_joint_sweeps": args.max_sweeps,
    }
    regression["smoke"] = args.smoke
    regression["candidate_budgets"] = {
        "medium": {"used": medium_count, "limit": args.medium_limit},
        "formal": {"used": formal_count, "limit": args.formal_limit},
    }
    written = write_results(
        output_dir=args.output,
        baseline=baseline,
        regression=regression,
        tower_rows=tower_rows,
        geometry_rows=geometry_rows,
        sensitivity_rows=sensitivity_rows,
        active_payload=active_payload,
        search_trace=search.trace,
        formal_rows=formal_rows,
        baseline_formal=baseline_formal,
        attempted_formal=attempted_formal,
        selected_formal=selected_formal,
        selected_design=selected_design,
        dense_payload=dense_payload,
        result3_template=args.result3_template,
        target_power_mw=args.target_power,
        decision=decision,
    )
    figures = generate_figures(
        sensitivity_rows=sensitivity_rows,
        tower_rows=tower_rows,
        geometry_rows=geometry_rows,
        selected_tower_mode=current_design.tower_mode,
        baseline=baseline,
        selected_design=selected_design,
        baseline_formal=baseline_formal,
        candidate_formal=attempted_formal,
        dense_payload=dense_payload,
        output_dir=args.output,
    )
    for index, path in enumerate(figures, start=16):
        written[f"figure_{index}"] = path

    print("\n六区参数微调结果", flush=True)
    print(f"判定：{decision}", flush=True)
    print(f"正式候选：P={attempted_formal.annual_power_mw:.9f} MW", flush=True)
    print(f"正式候选：q={attempted_formal.unit_area_power_kw_m2:.9f} kW/m²", flush=True)
    print(f"候选预算：medium={medium_count}/{args.medium_limit}，formal={formal_count}/{args.formal_limit}", flush=True)
    for path in written.values():
        print(f"输出：{path}", flush=True)
    return 0


def main() -> None:
    raise SystemExit(run())
