"""Sobol 初值筛选与分块整批 best-improvement 搜索。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import qmc

from ..q2.evaluate import EvaluationProfile
from .evaluate import Campo2DEvaluation, EvaluationCache, evaluate_design
from .model import (
    Campo2DBase,
    Campo2DDesign,
    build_campo_field,
    expand_design,
    validate_heterogeneous_field,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class InitialScreenRecord:
    index: int
    source: str
    design: Campo2DDesign
    geometry_valid: bool
    mirror_count: int | None
    total_area_m2: float | None
    coarse_power_mw: float | None
    coarse_q_kw_m2: float | None
    retained: bool
    reason: str


@dataclass(frozen=True)
class SearchTraceRecord:
    sequence: int
    start_name: str
    joint_cycle: int
    phase: str
    block: str
    step: float
    round_index: int
    candidate_count: int
    legal_candidate_count: int
    medium_candidate_count: int
    action: str
    accepted: bool
    evaluation_profile: str
    previous_annual_power_mw: float
    previous_total_area_m2: float
    previous_unit_area_power_kw_m2: float
    previous_design: Campo2DDesign
    annual_power_mw: float
    total_area_m2: float
    unit_area_power_kw_m2: float
    power_margin_mw: float
    design: Campo2DDesign


@dataclass(frozen=True)
class StartOutcome:
    start_name: str
    initial_design: Campo2DDesign
    initial_evaluation: Campo2DEvaluation
    best_design: Campo2DDesign
    best_evaluation: Campo2DEvaluation
    joint_cycles: int
    stable_joint_cycles: int
    stopped_by: str
    trace: tuple[SearchTraceRecord, ...]


@dataclass(frozen=True)
class MultiStartOutcome:
    initial_screen: tuple[InitialScreenRecord, ...]
    starts: tuple[StartOutcome, ...]
    best_start_name: str
    best_design: Campo2DDesign
    best_evaluation: Campo2DEvaluation


def _design_vector(design: Campo2DDesign) -> np.ndarray:
    canonical = design.canonical()
    return np.asarray(
        (
            canonical.tower_y,
            canonical.initial_spacing,
            canonical.spacing_growth,
            float(canonical.ring_count),
            *canonical.size_nodes,
            *canonical.height_nodes,
            *canonical.size_angles,
            *canonical.height_angles,
            canonical.area_scale,
        ),
        dtype=float,
    )


def _design_key(design: Campo2DDesign) -> tuple[float, ...]:
    return tuple(round(value, 10) for value in _design_vector(design))


def _replace_value(
    values: tuple[float, ...],
    index: int,
    value: float,
) -> tuple[float, ...]:
    result = list(values)
    result[index] = value
    return tuple(result)


def _within_bounds(design: Campo2DDesign) -> bool:
    canonical = design.canonical()
    return (
        abs(canonical.tower_x) <= 1e-12
        and -195.0 <= canonical.tower_y <= -170.0
        and 11.1 <= canonical.initial_spacing <= 12.7
        and 0.08 <= canonical.spacing_growth <= 0.28
        and 24 <= canonical.ring_count <= 32
        and all(abs(value) <= 0.30 for value in canonical.size_nodes)
        and all(2.0 <= value <= 6.0 for value in canonical.height_nodes)
        and all(abs(value) <= 0.15 for value in canonical.size_angles)
        and all(abs(value) <= 1.0 for value in canonical.height_angles)
        and 0.90 <= canonical.area_scale <= 1.05
    )


def _geometry_valid(
    base: Campo2DBase,
    design: Campo2DDesign,
) -> tuple[bool, str]:
    if not _within_bounds(design):
        return False, "超出局部参数范围"
    try:
        field = build_campo_field(base, design)
        check = validate_heterogeneous_field(
            field=field,
            specifications=expand_design(field, design),
        )
    except ValueError as exc:
        return False, str(exc)
    return check.valid, check.reason or "合法"


def _sobol_designs(
    *,
    base: Campo2DBase,
    count: int,
    seed: int,
) -> tuple[Campo2DDesign, ...]:
    if count < 0:
        raise ValueError("Sobol 初值数量不能小于 0。")
    if count == 0:
        return ()
    sampler = qmc.Sobol(d=19, scramble=True, seed=seed)
    unit = sampler.random(count)
    lower = np.asarray(
        (
            -195.0,
            11.1,
            0.08,
            float(max(24, base.ring_count - 2)),
            *([-0.04] * 5),
            *([base.parameters.installation_height - 0.4] * 5),
            -0.03,
            -0.03,
            -0.25,
            -0.25,
            0.97,
        ),
        dtype=float,
    )
    upper = np.asarray(
        (
            -170.0,
            12.7,
            0.28,
            float(min(32, base.ring_count + 2)),
            *([0.04] * 5),
            *([base.parameters.installation_height + 0.4] * 5),
            0.03,
            0.03,
            0.25,
            0.25,
            1.02,
        ),
        dtype=float,
    )
    scaled = qmc.scale(unit, lower, upper)
    designs: list[Campo2DDesign] = []
    for row in scaled:
        designs.append(
            Campo2DDesign(
                tower_y=float(row[0]),
                initial_spacing=float(row[1]),
                spacing_growth=float(row[2]),
                ring_count=int(np.clip(np.rint(row[3]), 24, 32)),
                size_nodes=tuple(float(value) for value in row[4:9]),
                height_nodes=tuple(float(value) for value in row[9:14]),
                size_angles=(float(row[14]), float(row[15])),
                height_angles=(float(row[16]), float(row[17])),
                area_scale=float(row[18]),
            ).canonical()
        )
    return tuple(designs)


def _coarse_rank(
    evaluation: Campo2DEvaluation,
    *,
    target_power_mw: float,
) -> tuple[int, float, float]:
    feasible = evaluation.is_feasible(target_power_mw)
    return (
        int(feasible),
        evaluation.unit_area_power_kw_m2 if feasible else evaluation.annual_power_mw,
        -evaluation.total_area_m2,
    )


def screen_initial_designs(
    *,
    base: Campo2DBase,
    coarse_profile: EvaluationProfile,
    sobol_count: int,
    retained_count: int,
    target_power_mw: float = 42.0,
    seed: int = 2023,
    cache: EvaluationCache | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[tuple[InitialScreenRecord, ...], tuple[tuple[str, Campo2DDesign], ...]]:
    if retained_count < 1:
        raise ValueError("至少保留一个搜索起点。")
    working_cache = cache or EvaluationCache()
    uniform = Campo2DDesign.uniform(
        base.parameters,
        ring_count=base.ring_count,
    )
    requested = (("uniform", uniform),) + tuple(
        (f"sobol-{index:03d}", design)
        for index, design in enumerate(
            _sobol_designs(
                base=base,
                count=sobol_count,
                seed=seed,
            ),
            start=1,
        )
    )
    evaluated: list[tuple[int, str, Campo2DDesign, Campo2DEvaluation]] = []
    raw: list[tuple[int, str, Campo2DDesign, bool, str, Campo2DEvaluation | None]] = []
    for index, (source, design) in enumerate(requested):
        valid, reason = _geometry_valid(base, design)
        evaluation: Campo2DEvaluation | None = None
        if valid:
            try:
                evaluation = evaluate_design(
                    base=base,
                    design=design,
                    profile=coarse_profile,
                    cache=working_cache,
                )
            except ValueError as exc:
                valid = False
                reason = str(exc)
        if evaluation is not None:
            evaluated.append((index, source, design, evaluation))
            if progress is not None:
                progress(
                    f"初值 {source}：P={evaluation.annual_power_mw:.6f} MW，"
                    f"q={evaluation.unit_area_power_kw_m2:.6f} kW/m²"
                )
        raw.append((index, source, design, valid, reason, evaluation))

    if not evaluated:
        raise RuntimeError("统一规格与 Sobol 初值均未通过几何和粗精度评价。")
    ranked = sorted(
        evaluated,
        key=lambda item: _coarse_rank(item[3], target_power_mw=target_power_mw),
        reverse=True,
    )
    selected: list[tuple[int, str, Campo2DDesign, Campo2DEvaluation]] = []
    uniform_item = next((item for item in evaluated if item[1] == "uniform"), None)
    if uniform_item is not None:
        selected.append(uniform_item)
    ranges = np.asarray(
        (25.0, 1.6, 0.2, 8.0, *([0.08] * 5), *([0.8] * 5), 0.06, 0.06, 0.5, 0.5, 0.05),
        dtype=float,
    )
    for item in ranked:
        if any(item[0] == previous[0] for previous in selected):
            continue
        vector = _design_vector(item[2])
        if any(
            float(np.linalg.norm((vector - _design_vector(previous[2])) / ranges)) < 0.10
            for previous in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= retained_count:
            break
    for item in ranked:
        if len(selected) >= retained_count:
            break
        if not any(item[0] == previous[0] for previous in selected):
            selected.append(item)

    retained_indices = {item[0] for item in selected}
    records = tuple(
        InitialScreenRecord(
            index=index,
            source=source,
            design=design,
            geometry_valid=valid,
            mirror_count=(evaluation.mirror_count if evaluation is not None else None),
            total_area_m2=(
                evaluation.total_area_m2 if evaluation is not None else None
            ),
            coarse_power_mw=(evaluation.annual_power_mw if evaluation is not None else None),
            coarse_q_kw_m2=(evaluation.unit_area_power_kw_m2 if evaluation is not None else None),
            retained=index in retained_indices,
            reason=(
                "保留"
                if index in retained_indices
                else ("粗精度排序未进入保留集合" if evaluation is not None else reason)
            ),
        )
        for index, source, design, valid, reason, evaluation in raw
    )
    starts = tuple((source, design) for _, source, design, _ in selected)
    return records, starts


def radial_height_candidates(
    design: Campo2DDesign,
    step: float,
) -> tuple[tuple[str, Campo2DDesign], ...]:
    return tuple(
        (
            f"beta{node + 1}{direction * step:+g}",
            replace(
                design,
                height_nodes=_replace_value(
                    design.height_nodes,
                    node,
                    design.height_nodes[node] + direction * step,
                ),
            ),
        )
        for node in range(5)
        for direction in (-1.0, 1.0)
    )


def radial_size_candidates(
    design: Campo2DDesign,
    step: float,
) -> tuple[tuple[str, Campo2DDesign], ...]:
    return tuple(
        (
            f"alpha{node + 1}{direction * step:+g}",
            replace(
                design,
                size_nodes=_replace_value(
                    design.size_nodes,
                    node,
                    design.size_nodes[node] + direction * step,
                ),
            ).canonical(),
        )
        for node in range(5)
        for direction in (-1.0, 1.0)
    )


def angle_candidates(
    design: Campo2DDesign,
    *,
    target: str,
    step: float,
) -> tuple[tuple[str, Campo2DDesign], ...]:
    values = design.size_angles if target == "size" else design.height_angles
    candidates: list[tuple[str, Campo2DDesign]] = []
    for index in range(2):
        for direction in (-1.0, 1.0):
            updated = _replace_value(values, index, values[index] + direction * step)
            kwargs = {"size_angles": updated} if target == "size" else {"height_angles": updated}
            candidates.append(
                (
                    f"{'a' if target == 'size' else 'b'}{index + 1}{direction * step:+g}",
                    replace(design, **kwargs),
                )
            )
    return tuple(candidates)


def scalar_candidates(
    design: Campo2DDesign,
    *,
    parameter: str,
    step: float,
) -> tuple[tuple[str, Campo2DDesign], ...]:
    value = getattr(design, parameter)
    return tuple(
        (
            f"{parameter}{direction * step:+g}",
            replace(design, **{parameter: value + direction * step}),
        )
        for direction in (-1.0, 1.0)
    )


def ring_candidates(design: Campo2DDesign) -> tuple[tuple[str, Campo2DDesign], ...]:
    return tuple(
        (
            f"ring_count{direction:+d}",
            replace(design, ring_count=design.ring_count + direction),
        )
        for direction in (-1, 1)
    )


class _SearchContext:
    def __init__(
        self,
        *,
        start_name: str,
        base: Campo2DBase,
        initial_design: Campo2DDesign,
        coarse_profile: EvaluationProfile,
        medium_profile: EvaluationProfile,
        coarse_cache: EvaluationCache,
        medium_cache: EvaluationCache,
        target_power_mw: float,
        move_q_threshold: float,
        medium_candidate_limit: int,
        progress: ProgressCallback | None,
    ) -> None:
        self.start_name = start_name
        self.base = base
        self.coarse_profile = coarse_profile
        self.medium_profile = medium_profile
        self.coarse_cache = coarse_cache
        self.medium_cache = medium_cache
        self.target_power_mw = target_power_mw
        self.move_q_threshold = move_q_threshold
        self.medium_candidate_limit = medium_candidate_limit
        self.progress = progress
        self.sequence = 0
        self.trace: list[SearchTraceRecord] = []
        self.current_design = initial_design.canonical()
        self.current_evaluation = evaluate_design(
            base=base,
            design=self.current_design,
            profile=medium_profile,
            cache=medium_cache,
        )
        self.initial_evaluation = self.current_evaluation

    def _better(
        self,
        candidate: Campo2DEvaluation,
        reference: Campo2DEvaluation,
    ) -> bool:
        candidate_feasible = candidate.is_feasible(self.target_power_mw)
        reference_feasible = reference.is_feasible(self.target_power_mw)
        if candidate_feasible != reference_feasible:
            return candidate_feasible
        if candidate_feasible:
            return (
                candidate.unit_area_power_kw_m2
                > reference.unit_area_power_kw_m2 + self.move_q_threshold
            )
        return candidate.annual_power_mw > reference.annual_power_mw + 1e-6

    def try_block(
        self,
        *,
        joint_cycle: int,
        phase: str,
        block: str,
        step: float,
        round_index: int,
        candidates: Sequence[tuple[str, Campo2DDesign]],
    ) -> bool:
        previous_design = self.current_design
        previous_evaluation = self.current_evaluation
        unique: list[tuple[str, Campo2DDesign]] = []
        seen = {_design_key(self.current_design)}
        for action, design in candidates:
            canonical = design.canonical()
            key = _design_key(canonical)
            if key in seen:
                continue
            seen.add(key)
            unique.append((action, canonical))

        coarse_values: list[tuple[str, Campo2DDesign, Campo2DEvaluation]] = []
        legal_count = 0
        for action, design in unique:
            valid, _ = _geometry_valid(self.base, design)
            if not valid:
                continue
            legal_count += 1
            try:
                evaluation = evaluate_design(
                    base=self.base,
                    design=design,
                    profile=self.coarse_profile,
                    cache=self.coarse_cache,
                )
            except ValueError:
                continue
            coarse_values.append((action, design, evaluation))
        ranked = sorted(
            coarse_values,
            key=lambda item: _coarse_rank(item[2], target_power_mw=self.target_power_mw),
            reverse=True,
        )
        finalists = ranked[: self.medium_candidate_limit]
        medium_values: list[tuple[str, Campo2DDesign, Campo2DEvaluation]] = []
        for action, design, _ in finalists:
            try:
                evaluation = evaluate_design(
                    base=self.base,
                    design=design,
                    profile=self.medium_profile,
                    cache=self.medium_cache,
                )
            except ValueError:
                continue
            medium_values.append((action, design, evaluation))

        best: tuple[str, Campo2DDesign, Campo2DEvaluation] | None = None
        for item in medium_values:
            if best is None or self._better(item[2], best[2]):
                best = item
        accepted = best is not None and self._better(best[2], self.current_evaluation)
        if accepted and best is not None:
            self.current_design = best[1]
            self.current_evaluation = best[2]
        record_evaluation = best[2] if best is not None else self.current_evaluation
        record_design = best[1] if best is not None else self.current_design
        self.sequence += 1
        self.trace.append(
            SearchTraceRecord(
                sequence=self.sequence,
                start_name=self.start_name,
                joint_cycle=joint_cycle,
                phase=phase,
                block=block,
                step=step,
                round_index=round_index,
                candidate_count=len(unique),
                legal_candidate_count=legal_count,
                medium_candidate_count=len(medium_values),
                action=best[0] if best is not None else "no-legal-candidate",
                accepted=accepted,
                evaluation_profile=self.medium_profile.name,
                previous_annual_power_mw=previous_evaluation.annual_power_mw,
                previous_total_area_m2=previous_evaluation.total_area_m2,
                previous_unit_area_power_kw_m2=(
                    previous_evaluation.unit_area_power_kw_m2
                ),
                previous_design=previous_design,
                annual_power_mw=record_evaluation.annual_power_mw,
                total_area_m2=record_evaluation.total_area_m2,
                unit_area_power_kw_m2=record_evaluation.unit_area_power_kw_m2,
                power_margin_mw=record_evaluation.annual_power_mw - self.target_power_mw,
                design=record_design,
            )
        )
        if accepted and best is not None and self.progress is not None:
            self.progress(
                f"[{self.start_name}] {phase}/{block} 接受 {best[0]}："
                f"P={best[2].annual_power_mw:.6f} MW，"
                f"q={best[2].unit_area_power_kw_m2:.6f} kW/m²"
            )
        return accepted


def _converge_block(
    context: _SearchContext,
    *,
    joint_cycle: int,
    phase: str,
    block: str,
    steps: Iterable[float],
    candidate_factory: Callable[[Campo2DDesign, float], Sequence[tuple[str, Campo2DDesign]]],
    maximum_rounds: int,
) -> bool:
    improved_any = False
    for step in steps:
        for round_index in range(1, maximum_rounds + 1):
            improved = context.try_block(
                joint_cycle=joint_cycle,
                phase=phase,
                block=block,
                step=step,
                round_index=round_index,
                candidates=candidate_factory(context.current_design, step),
            )
            improved_any |= improved
            if not improved:
                break
    return improved_any


def optimize_start(
    *,
    start_name: str,
    initial_design: Campo2DDesign,
    base: Campo2DBase,
    coarse_profile: EvaluationProfile,
    medium_profile: EvaluationProfile,
    coarse_cache: EvaluationCache,
    medium_cache: EvaluationCache,
    target_power_mw: float,
    move_q_threshold: float,
    convergence_q_threshold: float,
    maximum_rounds: int,
    maximum_joint_cycles: int,
    medium_candidate_limit: int,
    progress: ProgressCallback | None,
) -> StartOutcome:
    context = _SearchContext(
        start_name=start_name,
        base=base,
        initial_design=initial_design,
        coarse_profile=coarse_profile,
        medium_profile=medium_profile,
        coarse_cache=coarse_cache,
        medium_cache=medium_cache,
        target_power_mw=target_power_mw,
        move_q_threshold=move_q_threshold,
        medium_candidate_limit=medium_candidate_limit,
        progress=progress,
    )

    _converge_block(
        context,
        joint_cycle=0,
        phase="radial",
        block="height",
        steps=(0.4, 0.2, 0.1, 0.05),
        candidate_factory=radial_height_candidates,
        maximum_rounds=maximum_rounds,
    )
    _converge_block(
        context,
        joint_cycle=0,
        phase="radial",
        block="size",
        steps=(0.04, 0.02, 0.01, 0.005),
        candidate_factory=radial_size_candidates,
        maximum_rounds=maximum_rounds,
    )
    _converge_block(
        context,
        joint_cycle=0,
        phase="radial",
        block="lambda",
        steps=(0.005, 0.001, 0.0002),
        candidate_factory=lambda design, step: scalar_candidates(
            design,
            parameter="area_scale",
            step=step,
        ),
        maximum_rounds=maximum_rounds,
    )
    _converge_block(
        context,
        joint_cycle=0,
        phase="angular",
        block="size-angle",
        steps=(0.02, 0.01, 0.005),
        candidate_factory=lambda design, step: angle_candidates(
            design,
            target="size",
            step=step,
        ),
        maximum_rounds=maximum_rounds,
    )
    _converge_block(
        context,
        joint_cycle=0,
        phase="angular",
        block="height-angle",
        steps=(0.2, 0.1, 0.05),
        candidate_factory=lambda design, step: angle_candidates(
            design,
            target="height",
            step=step,
        ),
        maximum_rounds=maximum_rounds,
    )
    for parameter, steps in (
        ("tower_y", (4.0, 2.0, 1.0, 0.5)),
        ("initial_spacing", (0.4, 0.2, 0.1, 0.05)),
        ("spacing_growth", (0.04, 0.02, 0.01, 0.005)),
    ):
        _converge_block(
            context,
            joint_cycle=0,
            phase="geometry",
            block=parameter,
            steps=steps,
            candidate_factory=lambda design, step, parameter=parameter: scalar_candidates(
                design,
                parameter=parameter,
                step=step,
            ),
            maximum_rounds=maximum_rounds,
        )
    _converge_block(
        context,
        joint_cycle=0,
        phase="geometry",
        block="ring-count",
        steps=(1.0,),
        candidate_factory=lambda design, _step: ring_candidates(design),
        maximum_rounds=maximum_rounds,
    )

    stable_cycles = 0
    joint_cycles = 0
    stopped_by = "maximum_joint_cycles"
    joint_blocks = (
        ("radial-size", 0.005, radial_size_candidates),
        ("radial-height", 0.05, radial_height_candidates),
        (
            "size-angle",
            0.005,
            lambda design, step: angle_candidates(design, target="size", step=step),
        ),
        (
            "height-angle",
            0.05,
            lambda design, step: angle_candidates(design, target="height", step=step),
        ),
        (
            "lambda",
            0.0002,
            lambda design, step: scalar_candidates(
                design,
                parameter="area_scale",
                step=step,
            ),
        ),
        (
            "tower-y",
            0.5,
            lambda design, step: scalar_candidates(design, parameter="tower_y", step=step),
        ),
        (
            "initial-spacing",
            0.05,
            lambda design, step: scalar_candidates(
                design,
                parameter="initial_spacing",
                step=step,
            ),
        ),
        (
            "spacing-growth",
            0.005,
            lambda design, step: scalar_candidates(
                design,
                parameter="spacing_growth",
                step=step,
            ),
        ),
        ("ring-count", 1.0, lambda design, _step: ring_candidates(design)),
    )
    for cycle in range(1, maximum_joint_cycles + 1):
        joint_cycles = cycle
        before = context.current_evaluation.unit_area_power_kw_m2
        for block, step, factory in joint_blocks:
            _converge_block(
                context,
                joint_cycle=cycle,
                phase="joint",
                block=block,
                steps=(step,),
                candidate_factory=factory,
                maximum_rounds=maximum_rounds,
            )
        improvement = context.current_evaluation.unit_area_power_kw_m2 - before
        stable_cycles = stable_cycles + 1 if improvement < convergence_q_threshold else 0
        if progress is not None:
            progress(
                f"[{start_name}] 联合循环 {cycle}：Δq={improvement:.8f}，"
                f"连续稳定轮数={stable_cycles}"
            )
        if stable_cycles >= 2:
            stopped_by = "two_stable_joint_cycles"
            break

    return StartOutcome(
        start_name=start_name,
        initial_design=initial_design,
        initial_evaluation=context.initial_evaluation,
        best_design=context.current_design,
        best_evaluation=context.current_evaluation,
        joint_cycles=joint_cycles,
        stable_joint_cycles=stable_cycles,
        stopped_by=stopped_by,
        trace=tuple(context.trace),
    )


def optimize_multi_start(
    *,
    base: Campo2DBase,
    coarse_profile: EvaluationProfile,
    medium_profile: EvaluationProfile,
    sobol_count: int = 16,
    retained_count: int = 3,
    target_power_mw: float = 42.0,
    move_q_threshold: float = 1e-5,
    convergence_q_threshold: float = 1e-5,
    maximum_rounds: int = 4,
    maximum_joint_cycles: int = 6,
    medium_candidate_limit: int = 4,
    seed: int = 2023,
    progress: ProgressCallback | None = None,
) -> MultiStartOutcome:
    coarse_cache = EvaluationCache()
    medium_cache = EvaluationCache()
    screen, starts = screen_initial_designs(
        base=base,
        coarse_profile=coarse_profile,
        sobol_count=sobol_count,
        retained_count=retained_count,
        target_power_mw=target_power_mw,
        seed=seed,
        cache=coarse_cache,
        progress=progress,
    )
    outcomes = tuple(
        optimize_start(
            start_name=name,
            initial_design=design,
            base=base,
            coarse_profile=coarse_profile,
            medium_profile=medium_profile,
            coarse_cache=coarse_cache,
            medium_cache=medium_cache,
            target_power_mw=target_power_mw,
            move_q_threshold=move_q_threshold,
            convergence_q_threshold=convergence_q_threshold,
            maximum_rounds=maximum_rounds,
            maximum_joint_cycles=maximum_joint_cycles,
            medium_candidate_limit=medium_candidate_limit,
            progress=progress,
        )
        for name, design in starts
    )
    feasible = [
        outcome for outcome in outcomes if outcome.best_evaluation.is_feasible(target_power_mw)
    ]
    if not feasible:
        raise RuntimeError("所有保留初值均未得到满足 42 MW 的中精度候选。")
    best = max(feasible, key=lambda outcome: outcome.best_evaluation.unit_area_power_kw_m2)
    return MultiStartOutcome(
        initial_screen=screen,
        starts=outcomes,
        best_start_name=best.start_name,
        best_design=best.best_design,
        best_evaluation=best.best_evaluation,
    )
