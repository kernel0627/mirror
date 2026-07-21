"""最终候选的塔位包围扫描和正式精度最细邻域收口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .evaluate import RefineEvaluation
from .model import RefineDesign


Evaluator = Callable[[RefineDesign], RefineEvaluation | None]

LOCAL_STEPS = (
    ("tower_y", 0.1),
    ("initial_spacing", 0.05),
    ("spacing_growth", 0.005),
    ("w1", 0.02),
    ("h1", 0.02),
    ("H2", 0.02),
)


@dataclass(frozen=True)
class ClosureOutcome:
    initial_design: RefineDesign
    initial_evaluation: RefineEvaluation
    best_design: RefineDesign
    best_evaluation: RefineEvaluation
    trace: tuple[dict[str, object], ...]
    tower_bracketed: bool
    local_converged: bool
    local_sweeps: int


def _score(
    evaluation: RefineEvaluation,
    target_power_mw: float,
) -> tuple[int, float]:
    feasible = evaluation.is_feasible(target_power_mw)
    return (
        int(feasible),
        (
            evaluation.unit_area_power_kw_m2
            if feasible
            else evaluation.annual_power_mw
        ),
    )


def _better(
    candidate: RefineEvaluation,
    reference: RefineEvaluation,
    *,
    target_power_mw: float,
    threshold: float,
) -> bool:
    if candidate.is_feasible(target_power_mw) != reference.is_feasible(
        target_power_mw
    ):
        return candidate.is_feasible(target_power_mw)
    if candidate.is_feasible(target_power_mw):
        return (
            candidate.unit_area_power_kw_m2
            > reference.unit_area_power_kw_m2 + threshold
        )
    return candidate.annual_power_mw > reference.annual_power_mw + 1e-6


def _record(
    *,
    phase: str,
    sweep: int,
    parameter: str,
    old_value: float,
    new_value: float,
    design: RefineDesign,
    evaluation: RefineEvaluation | None,
    target_power_mw: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "phase": phase,
        "sweep": sweep,
        "parameter": parameter,
        "old_value": old_value,
        "new_value": new_value,
        "step": new_value - old_value,
        "tower_y": design.tower_y,
        "initial_spacing": design.initial_spacing,
        "spacing_growth": design.spacing_growth,
        "w1": design.widths[0],
        "h1": design.mirror_heights[0],
        "H2": design.installation_heights[1],
        "legal": evaluation is not None,
        "feasible": False,
        "annual_power_mw": None,
        "unit_area_power_kw_m2": None,
        "accepted": False,
        "stage_selected": False,
    }
    if evaluation is not None:
        row.update(
            {
                "feasible": evaluation.is_feasible(target_power_mw),
                "annual_power_mw": evaluation.annual_power_mw,
                "unit_area_power_kw_m2": (
                    evaluation.unit_area_power_kw_m2
                ),
            }
        )
    return row


def close_formal_neighborhood(
    *,
    initial_design: RefineDesign,
    initial_evaluation: RefineEvaluation,
    evaluator: Evaluator,
    target_power_mw: float = 42.0,
    coarse_step_limit: int = 12,
    fine_radius_steps: int = 4,
    maximum_local_sweeps: int = 4,
    move_q_threshold: float = 1e-8,
) -> ClosureOutcome:
    """先包围塔位极值，再对六个活跃变量做正式精度双侧检查。"""

    if coarse_step_limit < 1:
        raise ValueError("塔位包围扫描至少需要一个 0.5 m 步长。")
    if fine_radius_steps < 1:
        raise ValueError("塔位细扫至少需要中心两侧各一个 0.1 m 点。")
    if maximum_local_sweeps < 0:
        raise ValueError("最细邻域回扫轮数不能为负。")

    trace: list[dict[str, object]] = []
    coarse: list[tuple[RefineDesign, RefineEvaluation, dict[str, object] | None]] = [
        (initial_design, initial_evaluation, None)
    ]
    previous = initial_evaluation
    tower_bracketed = False
    for index in range(1, coarse_step_limit + 1):
        candidate = initial_design.with_parameter(
            "tower_y",
            initial_design.tower_y + 0.5 * index,
        )
        evaluation = evaluator(candidate)
        row = _record(
            phase="tower_coarse",
            sweep=0,
            parameter="tower_y",
            old_value=initial_design.tower_y,
            new_value=candidate.tower_y,
            design=candidate,
            evaluation=evaluation,
            target_power_mw=target_power_mw,
        )
        trace.append(row)
        if evaluation is None:
            tower_bracketed = True
            break
        coarse.append((candidate, evaluation, row))
        if (
            previous.is_feasible(target_power_mw)
            and evaluation.is_feasible(target_power_mw)
            and evaluation.unit_area_power_kw_m2
            < previous.unit_area_power_kw_m2
        ):
            tower_bracketed = True
            break
        previous = evaluation

    coarse_best = max(
        coarse,
        key=lambda item: _score(item[1], target_power_mw),
    )
    if coarse_best[2] is not None:
        coarse_best[2]["stage_selected"] = True

    fine: list[tuple[RefineDesign, RefineEvaluation, dict[str, object]]] = []
    center_y = coarse_best[0].tower_y
    for offset in range(-fine_radius_steps, fine_radius_steps + 1):
        candidate = coarse_best[0].with_parameter(
            "tower_y",
            center_y + 0.1 * offset,
        )
        evaluation = evaluator(candidate)
        row = _record(
            phase="tower_fine",
            sweep=0,
            parameter="tower_y",
            old_value=center_y,
            new_value=candidate.tower_y,
            design=candidate,
            evaluation=evaluation,
            target_power_mw=target_power_mw,
        )
        trace.append(row)
        if evaluation is not None:
            fine.append((candidate, evaluation, row))
    fine_best = max(
        fine,
        key=lambda item: _score(item[1], target_power_mw),
    )
    fine_best[2]["stage_selected"] = True
    current_design = fine_best[0]
    current_evaluation = fine_best[1]

    local_converged = maximum_local_sweeps == 0
    completed_sweeps = 0
    for sweep in range(1, maximum_local_sweeps + 1):
        completed_sweeps = sweep
        sweep_improved = False
        for parameter, step in LOCAL_STEPS:
            old_value = current_design.parameter(parameter)
            candidates: list[
                tuple[RefineDesign, RefineEvaluation, dict[str, object]]
            ] = []
            for sign in (-1.0, 1.0):
                candidate = current_design.with_parameter(
                    parameter,
                    old_value + sign * step,
                )
                evaluation = evaluator(candidate)
                row = _record(
                    phase="local_fine",
                    sweep=sweep,
                    parameter=parameter,
                    old_value=old_value,
                    new_value=candidate.parameter(parameter),
                    design=candidate,
                    evaluation=evaluation,
                    target_power_mw=target_power_mw,
                )
                trace.append(row)
                if evaluation is not None:
                    candidates.append((candidate, evaluation, row))
            improving = [
                item
                for item in candidates
                if _better(
                    item[1],
                    current_evaluation,
                    target_power_mw=target_power_mw,
                    threshold=move_q_threshold,
                )
            ]
            if improving:
                best = max(
                    improving,
                    key=lambda item: _score(item[1], target_power_mw),
                )
                best[2]["accepted"] = True
                current_design = best[0]
                current_evaluation = best[1]
                sweep_improved = True
        if not sweep_improved:
            local_converged = True
            break

    return ClosureOutcome(
        initial_design=initial_design,
        initial_evaluation=initial_evaluation,
        best_design=current_design,
        best_evaluation=current_evaluation,
        trace=tuple(trace),
        tower_bracketed=tower_bracketed,
        local_converged=local_converged,
        local_sweeps=completed_sweeps,
    )
