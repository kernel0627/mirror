"""活跃变量的分块变步长局部搜索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .evaluate import RefineEvaluation
from .model import RefineDesign


Evaluator = Callable[[RefineDesign], RefineEvaluation | None]

BLOCK_ORDER = ("tower", "campo", "width", "height", "installation")
STEP_LEVELS = {
    "tower_y": (0.50, 0.25, 0.10),
    "initial_spacing": (0.20, 0.10, 0.05),
    "spacing_growth": (0.020, 0.010, 0.005),
    "w": (0.10, 0.05, 0.02),
    "h": (0.10, 0.05, 0.02),
    "H": (0.10, 0.05, 0.02),
}


@dataclass(frozen=True)
class SearchOutcome:
    initial_design: RefineDesign
    initial_evaluation: RefineEvaluation
    best_design: RefineDesign
    best_evaluation: RefineEvaluation
    trace: tuple[dict[str, object], ...]
    evaluated_candidates: int


def parameter_block(parameter: str) -> str:
    if parameter == "tower_y":
        return "tower"
    if parameter in ("initial_spacing", "spacing_growth"):
        return "campo"
    return {"w": "width", "h": "height", "H": "installation"}[parameter[0]]


def _steps(parameter: str) -> tuple[float, ...]:
    if parameter in STEP_LEVELS:
        return STEP_LEVELS[parameter]
    return STEP_LEVELS[parameter[0]]


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


def coordinate_search(
    *,
    initial_design: RefineDesign,
    initial_evaluation: RefineEvaluation,
    active_variables: tuple[str, ...],
    evaluator: Evaluator,
    baseline_q_kw_m2: float,
    maximum_sweeps: int = 2,
    target_power_mw: float = 42.0,
    move_q_threshold: float = 1e-8,
) -> SearchOutcome:
    """按塔位、Campo、宽、高、安装高顺序执行最多两轮回扫。"""

    if maximum_sweeps < 0 or maximum_sweeps > 2:
        raise ValueError("联合回扫轮数必须位于 0 到 2。")
    current_design = initial_design
    current_evaluation = initial_evaluation
    level_by_block = {block: 0 for block in BLOCK_ORDER}
    trace: list[dict[str, object]] = []
    evaluated = 0

    for sweep in range(1, maximum_sweeps + 1):
        sweep_improved = False
        for block in BLOCK_ORDER:
            parameters = tuple(
                parameter
                for parameter in active_variables
                if parameter_block(parameter) == block
            )
            if not parameters:
                continue
            level = level_by_block[block]
            ranked: list[tuple[str, float, float, RefineDesign, RefineEvaluation]] = []
            for parameter in parameters:
                step = _steps(parameter)[level]
                old = current_design.parameter(parameter)
                for sign in (-1.0, 1.0):
                    new = old + sign * step
                    candidate_design = current_design.with_parameter(parameter, new)
                    evaluation = evaluator(candidate_design)
                    if evaluation is None:
                        continue
                    evaluated += 1
                    ranked.append((parameter, old, new, candidate_design, evaluation))
            improving = [
                item
                for item in ranked
                if _better(
                    item[4],
                    current_evaluation,
                    target_power_mw=target_power_mw,
                    threshold=move_q_threshold,
                )
            ]
            if improving:
                parameter, old, new, design, evaluation = max(
                    improving,
                    key=lambda item: (
                        int(item[4].is_feasible(target_power_mw)),
                        item[4].unit_area_power_kw_m2
                        if item[4].is_feasible(target_power_mw)
                        else item[4].annual_power_mw,
                    ),
                )
                previous = current_evaluation
                current_design = design
                current_evaluation = evaluation
                sweep_improved = True
                trace.append(
                    {
                        "sweep_id": sweep,
                        "parameter_block": block,
                        "parameter": parameter,
                        "old_value": old,
                        "new_value": new,
                        "step_size": abs(new - old),
                        "evaluation_level": evaluation.profile_name,
                        "power": evaluation.annual_power_mw,
                        "power_margin": evaluation.annual_power_mw - target_power_mw,
                        "total_area": evaluation.total_area_m2,
                        "q": evaluation.unit_area_power_kw_m2,
                        "delta_q_from_previous": (
                            evaluation.unit_area_power_kw_m2
                            - previous.unit_area_power_kw_m2
                        ),
                        "delta_q_from_six": (
                            evaluation.unit_area_power_kw_m2
                            - baseline_q_kw_m2
                        ),
                        "accepted": True,
                    }
                )
            else:
                level_by_block[block] = min(level + 1, 2)
        if not sweep_improved:
            break

    return SearchOutcome(
        initial_design=initial_design,
        initial_evaluation=initial_evaluation,
        best_design=current_design,
        best_evaluation=current_evaluation,
        trace=tuple(trace),
        evaluated_candidates=evaluated,
    )
