"""五节点径向样条的三初值中精度收敛搜索。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    evaluate_design,
)
from .model import CampoMotherField, SplineDesign
from .model import (
    expand_spline_design,
    validate_heterogeneous_field,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SearchTraceRecord:
    sequence: int
    start_name: str
    phase: str
    step: float
    round_index: int
    action: str
    accepted: bool
    feasible: bool
    annual_power_mw: float
    total_area_m2: float
    unit_area_power_kw_m2: float
    size_nodes: tuple[float, ...]
    height_nodes: tuple[float, ...]
    area_scale: float


@dataclass(frozen=True)
class StartOutcome:
    start_name: str
    requested_initial_design: SplineDesign
    height_projection_factor: float
    size_projection_factor: float
    initial_evaluation: HeterogeneousEvaluation
    best_design: SplineDesign
    best_evaluation: HeterogeneousEvaluation
    joint_cycles: int
    stable_joint_cycles: int
    stopped_by: str
    trace: tuple[SearchTraceRecord, ...]


@dataclass(frozen=True)
class MultiStartOutcome:
    starts: tuple[StartOutcome, ...]
    best_start_name: str
    best_design: SplineDesign
    best_evaluation: HeterogeneousEvaluation


def _replace_node(
    values: tuple[float, ...],
    index: int,
    value: float,
) -> tuple[float, ...]:
    mutable = list(values)
    mutable[index] = value
    return tuple(mutable)


def _replace_two_nodes(
    values: tuple[float, ...],
    left: int,
    left_value: float,
    right: int,
    right_value: float,
) -> tuple[float, ...]:
    mutable = list(values)
    mutable[left] = left_value
    mutable[right] = right_value
    return tuple(mutable)


def height_candidates(
    design: SplineDesign,
    step: float,
) -> tuple[tuple[str, SplineDesign], ...]:
    candidates: list[tuple[str, SplineDesign]] = []
    for node in range(5):
        for direction in (-1.0, 1.0):
            heights = _replace_node(
                design.height_nodes,
                node,
                design.height_nodes[node] + direction * step,
            )
            candidates.append(
                (
                    f"beta{node + 1}{'+' if direction > 0 else '-'}{step:g}",
                    SplineDesign(
                        design.size_nodes,
                        heights,
                        design.area_scale,
                    ),
                )
            )
    for node in range(4):
        for direction in (-1.0, 1.0):
            heights = _replace_two_nodes(
                design.height_nodes,
                node,
                design.height_nodes[node] + direction * step,
                node + 1,
                design.height_nodes[node + 1] + direction * step,
            )
            candidates.append(
                (
                    f"beta{node + 1},{node + 2}"
                    f"{'+' if direction > 0 else '-'}{step:g}",
                    SplineDesign(
                        design.size_nodes,
                        heights,
                        design.area_scale,
                    ),
                )
            )
    return tuple(candidates)


def size_candidates(
    design: SplineDesign,
    step: float,
) -> tuple[tuple[str, SplineDesign], ...]:
    candidates: list[tuple[str, SplineDesign]] = []
    for node in range(5):
        for direction in (-1.0, 1.0):
            sizes = _replace_node(
                design.size_nodes,
                node,
                design.size_nodes[node] + direction * step,
            )
            candidates.append(
                (
                    f"alpha{node + 1}{'+' if direction > 0 else '-'}{step:g}",
                    SplineDesign(
                        sizes,
                        design.height_nodes,
                        design.area_scale,
                    ).canonical(),
                )
            )
    for left in range(5):
        for right in range(left + 1, 5):
            relation = "adjacent" if right - left == 1 else "cross"
            for direction in (-1.0, 1.0):
                sizes = _replace_two_nodes(
                    design.size_nodes,
                    left,
                    design.size_nodes[left] + direction * step,
                    right,
                    design.size_nodes[right] - direction * step,
                )
                candidates.append(
                    (
                        f"{relation}-alpha{left + 1}->alpha{right + 1}"
                        f"{'+' if direction > 0 else '-'}{step:g}",
                        SplineDesign(
                            sizes,
                            design.height_nodes,
                            design.area_scale,
                        ).canonical(),
                    )
                )
    return tuple(candidates)


def lambda_candidates(
    design: SplineDesign,
    values: Iterable[float],
) -> tuple[tuple[str, SplineDesign], ...]:
    return tuple(
        (
            f"lambda={value:.6f}",
            SplineDesign(
                design.size_nodes,
                design.height_nodes,
                float(value),
            ),
        )
        for value in values
    )


def _design_key(design: SplineDesign) -> tuple[float, ...]:
    canonical = design.canonical()
    return tuple(
        round(value, 10)
        for value in (
            *canonical.size_nodes,
            *canonical.height_nodes,
            canonical.area_scale,
        )
    )


def _within_parameter_bounds(design: SplineDesign) -> bool:
    canonical = design.canonical()
    return (
        all(abs(value) <= 0.35 for value in canonical.size_nodes)
        and all(2.0 <= value <= 6.0 for value in canonical.height_nodes)
        and 0.90 <= canonical.area_scale <= 1.05
    )


def _geometry_is_valid(
    mother: CampoMotherField,
    design: SplineDesign,
) -> bool:
    specifications = expand_spline_design(mother, design)
    return validate_heterogeneous_field(
        coordinates=mother.coordinates,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        tower_x=mother.parameters.tower_x,
        tower_y=mother.parameters.tower_y,
        field_radius=mother.parameters.field_radius,
        exclusion_radius=mother.parameters.exclusion_radius,
        safety_epsilon=mother.parameters.safety_epsilon,
    ).valid


def _largest_legal_factor(
    *,
    mother: CampoMotherField,
    design_at_zero: SplineDesign,
    design_at_one: SplineDesign,
) -> tuple[SplineDesign, float]:
    """沿给定初值方向保留尽可能大的合法幅度。"""

    zero = design_at_zero.canonical()
    one = design_at_one.canonical()

    def interpolate(factor: float) -> SplineDesign:
        return SplineDesign(
            size_nodes=tuple(
                left + factor * (right - left)
                for left, right in zip(zero.size_nodes, one.size_nodes)
            ),
            height_nodes=tuple(
                left + factor * (right - left)
                for left, right in zip(
                    zero.height_nodes,
                    one.height_nodes,
                )
            ),
            area_scale=(
                zero.area_scale
                + factor * (one.area_scale - zero.area_scale)
            ),
        ).canonical()

    if not _geometry_is_valid(mother, zero):
        raise RuntimeError("初值缩幅的零点方案不满足固定镜场几何约束。")
    if _geometry_is_valid(mother, one):
        return one, 1.0

    low = 0.0
    high = 1.0
    for _ in range(50):
        midpoint = 0.5 * (low + high)
        if _geometry_is_valid(mother, interpolate(midpoint)):
            low = midpoint
        else:
            high = midpoint
    factor = max(0.0, low - 1e-10)
    projected = interpolate(factor)
    if not _geometry_is_valid(mother, projected):
        raise RuntimeError("初值缩幅后仍未进入固定镜场合法域。")
    return projected, factor


class _SearchContext:
    def __init__(
        self,
        *,
        start_name: str,
        mother: CampoMotherField,
        profile: EvaluationProfile,
        initial_design: SplineDesign,
        cache: EvaluationCache,
        target_power_mw: float,
        move_q_threshold: float,
        workers: int,
        progress: ProgressCallback | None,
    ) -> None:
        self.start_name = start_name
        self.mother = mother
        self.profile = profile
        self.cache = cache
        self.target_power_mw = target_power_mw
        self.move_q_threshold = move_q_threshold
        self.workers = workers
        self.progress = progress
        self.trace: list[SearchTraceRecord] = []
        self.sequence = 0
        self.current_design = initial_design.canonical()
        self.current_evaluation = evaluate_design(
            mother=mother,
            design=self.current_design,
            profile=profile,
            cache=cache,
        )
        self.initial_evaluation = self.current_evaluation

    def _record(
        self,
        *,
        phase: str,
        step: float,
        round_index: int,
        action: str,
        design: SplineDesign,
        evaluation: HeterogeneousEvaluation,
        accepted: bool,
    ) -> None:
        self.sequence += 1
        self.trace.append(
            SearchTraceRecord(
                sequence=self.sequence,
                start_name=self.start_name,
                phase=phase,
                step=step,
                round_index=round_index,
                action=action,
                accepted=accepted,
                feasible=evaluation.is_feasible(
                    self.target_power_mw
                ),
                annual_power_mw=evaluation.annual_power_mw,
                total_area_m2=evaluation.total_area_m2,
                unit_area_power_kw_m2=(
                    evaluation.unit_area_power_kw_m2
                ),
                size_nodes=design.size_nodes,
                height_nodes=design.height_nodes,
                area_scale=design.area_scale,
            )
        )

    def _is_improvement(
        self,
        evaluation: HeterogeneousEvaluation,
    ) -> bool:
        if not evaluation.is_feasible(self.target_power_mw):
            return False
        return (
            not self.current_evaluation.is_feasible(
                self.target_power_mw
            )
            or evaluation.unit_area_power_kw_m2
            > self.current_evaluation.unit_area_power_kw_m2
            + self.move_q_threshold
        )

    def _evaluate_one(
        self,
        item: tuple[str, SplineDesign],
    ) -> tuple[str, SplineDesign, HeterogeneousEvaluation | None]:
        action, design = item
        canonical = design.canonical()
        if not _within_parameter_bounds(canonical):
            return action, canonical, None
        try:
            evaluation = evaluate_design(
                mother=self.mother,
                design=canonical,
                profile=self.profile,
                cache=self.cache,
            )
        except ValueError:
            evaluation = None
        return action, canonical, evaluation

    def _evaluate_batch(
        self,
        candidates: Sequence[tuple[str, SplineDesign]],
    ) -> list[
        tuple[str, SplineDesign, HeterogeneousEvaluation | None]
    ]:
        unique: list[tuple[str, SplineDesign]] = []
        seen: set[tuple[float, ...]] = set()
        for action, design in candidates:
            key = _design_key(design)
            if key in seen or key == _design_key(self.current_design):
                continue
            seen.add(key)
            unique.append((action, design))
        if self.workers == 1 or len(unique) <= 1:
            return [self._evaluate_one(item) for item in unique]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(self._evaluate_one, unique))

    def try_best_move(
        self,
        *,
        phase: str,
        step: float,
        round_index: int,
        candidates: Sequence[tuple[str, SplineDesign]],
    ) -> bool:
        evaluated = self._evaluate_batch(candidates)
        feasible = [
            item
            for item in evaluated
            if item[2] is not None
            and item[2].is_feasible(self.target_power_mw)
        ]
        best = max(
            feasible,
            key=lambda item: item[2].unit_area_power_kw_m2,
            default=None,
        )
        accepted = False
        if best is not None:
            best_evaluation = best[2]
            assert best_evaluation is not None
            accepted = (
                not self.current_evaluation.is_feasible(
                    self.target_power_mw
                )
                or best_evaluation.unit_area_power_kw_m2
                > self.current_evaluation.unit_area_power_kw_m2
                + self.move_q_threshold
            )

        for action, design, evaluation in evaluated:
            if evaluation is None:
                continue
            is_accepted = bool(
                accepted
                and best is not None
                and _design_key(design) == _design_key(best[1])
            )
            self._record(
                phase=phase,
                step=step,
                round_index=round_index,
                action=action,
                design=design,
                evaluation=evaluation,
                accepted=is_accepted,
            )
        if not accepted or best is None or best[2] is None:
            return False
        self.current_design = best[1]
        self.current_evaluation = best[2]
        if self.progress is not None:
            self.progress(
                f"[{self.start_name}] {phase} 接受 {best[0]}："
                f"P={best[2].annual_power_mw:.6f} MW，"
                f"q={best[2].unit_area_power_kw_m2:.6f} kW/m²"
            )
        return True

    def try_first_move(
        self,
        *,
        phase: str,
        step: float,
        round_index: int,
        candidates: Sequence[tuple[str, SplineDesign]],
    ) -> bool:
        ordered: Iterable[tuple[str, SplineDesign]] = (
            candidates if round_index % 2 == 1 else reversed(candidates)
        )
        seen: set[tuple[float, ...]] = set()
        for action, design in ordered:
            canonical = design.canonical()
            key = _design_key(canonical)
            if key in seen or key == _design_key(self.current_design):
                continue
            seen.add(key)
            _, _, evaluation = self._evaluate_one((action, canonical))
            if evaluation is None:
                continue
            accepted = self._is_improvement(evaluation)
            self._record(
                phase=phase,
                step=step,
                round_index=round_index,
                action=action,
                design=canonical,
                evaluation=evaluation,
                accepted=accepted,
            )
            if not accepted:
                continue
            self.current_design = canonical
            self.current_evaluation = evaluation
            if self.progress is not None:
                self.progress(
                    f"[{self.start_name}] {phase} 接受 {action}："
                    f"P={evaluation.annual_power_mw:.6f} MW，"
                    f"q={evaluation.unit_area_power_kw_m2:.6f} kW/m²"
                )
            return True
        return False


def _converge_height_step(
    context: _SearchContext,
    *,
    phase: str,
    step: float,
    maximum_rounds: int,
) -> bool:
    misses = 0
    improved_any = False
    for round_index in range(1, maximum_rounds + 1):
        improved = context.try_first_move(
            phase=phase,
            step=step,
            round_index=round_index,
            candidates=height_candidates(context.current_design, step),
        )
        improved_any |= improved
        misses = 0 if improved else misses + 1
        if misses >= 2:
            return improved_any
    raise RuntimeError(
        f"{phase} 在步长 {step:g} 下达到 {maximum_rounds} 轮仍未收敛。"
    )


def _converge_size_step(
    context: _SearchContext,
    *,
    phase: str,
    step: float,
    maximum_rounds: int,
    height_rescan_step: float,
) -> bool:
    misses = 0
    improved_any = False
    for round_index in range(1, maximum_rounds + 1):
        improved = context.try_first_move(
            phase=phase,
            step=step,
            round_index=round_index,
            candidates=size_candidates(context.current_design, step),
        )
        improved_any |= improved
        if improved:
            context.try_first_move(
                phase=f"{phase}-height-rescan",
                step=height_rescan_step,
                round_index=round_index,
                candidates=height_candidates(
                    context.current_design,
                    height_rescan_step,
                ),
            )
            misses = 0
        else:
            misses += 1
        if misses >= 2:
            return improved_any
    raise RuntimeError(
        f"{phase} 在步长 {step:g} 下达到 {maximum_rounds} 轮仍未收敛。"
    )


def _lambda_grid(
    center: float | None,
    *,
    step: float,
) -> tuple[float, ...]:
    if center is None:
        start, stop = 0.94, 1.02
    elif step == 0.001:
        start, stop = center - 0.005, center + 0.005
    else:
        start, stop = center - 0.001, center + 0.001
    count = int(round((stop - start) / step))
    return tuple(
        round(start + index * step, 10)
        for index in range(count + 1)
    )


def search_lambda(
    context: _SearchContext,
    *,
    phase: str,
) -> None:
    center: float | None = None
    for level, step in enumerate((0.005, 0.001, 0.0002), start=1):
        values = _lambda_grid(center, step=step)
        context.try_best_move(
            phase=f"{phase}-L{level}",
            step=step,
            round_index=1,
            candidates=lambda_candidates(context.current_design, values),
        )
        center = context.current_design.area_scale


def optimize_one_start(
    *,
    start_name: str,
    mother: CampoMotherField,
    initial_design: SplineDesign,
    profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    move_q_threshold: float = 1e-7,
    convergence_q_threshold: float = 1e-5,
    maximum_joint_cycles: int = 8,
    maximum_rounds_per_step: int = 40,
    workers: int = 1,
    cache: EvaluationCache | None = None,
    progress: ProgressCallback | None = None,
) -> StartOutcome:
    if workers < 1:
        raise ValueError("workers 必须大于等于 1。")
    working_cache = cache or EvaluationCache()
    original_initial = initial_design.canonical()
    uniform_initial = SplineDesign.uniform(
        mother.base_installation_height
    )
    requested_height_initial = SplineDesign(
        size_nodes=(0.0,) * 5,
        height_nodes=original_initial.height_nodes,
        area_scale=1.0,
    )
    height_initial, height_projection_factor = _largest_legal_factor(
        mother=mother,
        design_at_zero=uniform_initial,
        design_at_one=requested_height_initial,
    )
    if progress is not None and height_projection_factor < 1.0:
        progress(
            f"[{start_name}] 初始高度曲线按 "
            f"{height_projection_factor:.6f} 缩幅后进入合法域"
        )
    context = _SearchContext(
        start_name=start_name,
        mother=mother,
        profile=profile,
        initial_design=height_initial,
        cache=working_cache,
        target_power_mw=target_power_mw,
        move_q_threshold=move_q_threshold,
        workers=workers,
        progress=progress,
    )
    initial_evaluation = context.initial_evaluation

    for step in (0.4, 0.2, 0.1, 0.05):
        _converge_height_step(
            context,
            phase="height",
            step=step,
            maximum_rounds=maximum_rounds_per_step,
        )

    zero_size = SplineDesign(
        size_nodes=(0.0,) * 5,
        height_nodes=context.current_design.height_nodes,
        area_scale=1.0,
    )
    requested_size = SplineDesign(
        size_nodes=original_initial.size_nodes,
        height_nodes=context.current_design.height_nodes,
        area_scale=1.0,
    )
    (
        context.current_design,
        size_projection_factor,
    ) = _largest_legal_factor(
        mother=mother,
        design_at_zero=zero_size,
        design_at_one=requested_size,
    )
    context.current_evaluation = evaluate_design(
        mother=mother,
        design=context.current_design,
        profile=profile,
        cache=working_cache,
    )
    if progress is not None and size_projection_factor < 1.0:
        progress(
            f"[{start_name}] 初始尺寸形状按 "
            f"{size_projection_factor:.6f} "
            "缩幅后进入 lambda=1 合法域"
        )
    for step in (0.04, 0.02, 0.01, 0.005):
        _converge_size_step(
            context,
            phase="fixed-lambda-size",
            step=step,
            maximum_rounds=maximum_rounds_per_step,
            height_rescan_step=0.05,
        )

    search_lambda(context, phase="lambda")

    stable_cycles = 0
    joint_cycles = 0
    stopped_by = "maximum_joint_cycles"
    for cycle in range(1, maximum_joint_cycles + 1):
        joint_cycles = cycle
        before_q = context.current_evaluation.unit_area_power_kw_m2
        _converge_size_step(
            context,
            phase=f"joint-{cycle}-size",
            step=0.005,
            maximum_rounds=maximum_rounds_per_step,
            height_rescan_step=0.05,
        )
        _converge_height_step(
            context,
            phase=f"joint-{cycle}-height",
            step=0.05,
            maximum_rounds=maximum_rounds_per_step,
        )
        search_lambda(context, phase=f"joint-{cycle}-lambda")
        improvement = (
            context.current_evaluation.unit_area_power_kw_m2 - before_q
        )
        stable_cycles = (
            stable_cycles + 1
            if improvement < convergence_q_threshold
            else 0
        )
        if progress is not None:
            progress(
                f"[{start_name}] 联合循环 {cycle}："
                f"Δq={improvement:.8f}，"
                f"连续稳定轮数={stable_cycles}"
            )
        if stable_cycles >= 2:
            stopped_by = "two_stable_joint_cycles"
            break

    return StartOutcome(
        start_name=start_name,
        requested_initial_design=original_initial,
        height_projection_factor=height_projection_factor,
        size_projection_factor=size_projection_factor,
        initial_evaluation=initial_evaluation,
        best_design=context.current_design,
        best_evaluation=context.current_evaluation,
        joint_cycles=joint_cycles,
        stable_joint_cycles=stable_cycles,
        stopped_by=stopped_by,
        trace=tuple(context.trace),
    )


def optimize_three_starts(
    *,
    mother: CampoMotherField,
    initial_designs: Sequence[tuple[str, SplineDesign]],
    profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    move_q_threshold: float = 1e-7,
    convergence_q_threshold: float = 1e-5,
    maximum_joint_cycles: int = 8,
    maximum_rounds_per_step: int = 40,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> MultiStartOutcome:
    if len(initial_designs) != 3:
        raise ValueError("必须恰好提供三个初始解。")
    cache = EvaluationCache()
    outcomes = tuple(
        optimize_one_start(
            start_name=name,
            mother=mother,
            initial_design=design,
            profile=profile,
            target_power_mw=target_power_mw,
            move_q_threshold=move_q_threshold,
            convergence_q_threshold=convergence_q_threshold,
            maximum_joint_cycles=maximum_joint_cycles,
            maximum_rounds_per_step=maximum_rounds_per_step,
            workers=workers,
            cache=cache,
            progress=progress,
        )
        for name, design in initial_designs
    )
    feasible = [
        outcome
        for outcome in outcomes
        if outcome.best_evaluation.is_feasible(target_power_mw)
    ]
    if not feasible:
        raise RuntimeError("三个初始解均未得到满足 42 MW 的中精度方案。")
    best = max(
        feasible,
        key=lambda outcome: outcome.best_evaluation.unit_area_power_kw_m2,
    )
    return MultiStartOutcome(
        starts=outcomes,
        best_start_name=best.start_name,
        best_design=best.best_design,
        best_evaluation=best.best_evaluation,
    )
