"""第三问六组高度、面积再分配和面积压缩搜索。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    PowerCalibration,
    build_power_calibration,
    evaluate_design,
)
from .model import (
    GROUP_COUNT,
    CampoMotherField,
    GroupDesign,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SearchStep:
    stage: str
    action: str
    design: GroupDesign
    evaluation: HeterogeneousEvaluation
    estimated_power_mw: float
    empirical_bound_mw: float


@dataclass(frozen=True)
class SearchOutcome:
    baseline_design: GroupDesign
    baseline_evaluation: HeterogeneousEvaluation
    best_design: GroupDesign
    best_evaluation: HeterogeneousEvaluation
    calibration: PowerCalibration
    calibration_pairs: tuple[
        tuple[HeterogeneousEvaluation, HeterogeneousEvaluation],
        ...,
    ]
    trace: tuple[SearchStep, ...]
    stage_evaluations: tuple[
        tuple[str, HeterogeneousEvaluation],
        ...,
    ]


def _replace_tuple(
    values: tuple[float, ...],
    index: int,
    value: float,
) -> tuple[float, ...]:
    mutable = list(values)
    mutable[index] = value
    return tuple(mutable)


def with_height(
    design: GroupDesign,
    group: int,
    value: float,
) -> GroupDesign:
    return GroupDesign(
        scales=design.scales,
        heights=_replace_tuple(design.heights, group, value),
    )


def with_scale(
    design: GroupDesign,
    group: int,
    value: float,
) -> GroupDesign:
    return GroupDesign(
        scales=_replace_tuple(design.scales, group, value),
        heights=design.heights,
    )


def transfer_area(
    *,
    design: GroupDesign,
    source_group: int,
    target_group: int,
    delta_area_m2: float,
    group_counts: tuple[int, ...],
    base_mirror_area_m2: float,
) -> GroupDesign:
    if source_group == target_group:
        raise ValueError("面积转移的来源组和目标组不能相同。")
    if delta_area_m2 <= 0.0:
        raise ValueError("面积转移量必须大于 0。")
    source_square = (
        design.scales[source_group] ** 2
        - delta_area_m2
        / (group_counts[source_group] * base_mirror_area_m2)
    )
    target_square = (
        design.scales[target_group] ** 2
        + delta_area_m2
        / (group_counts[target_group] * base_mirror_area_m2)
    )
    if source_square <= 0.0:
        raise ValueError("面积转移量超过来源组当前面积。")
    scales = list(design.scales)
    scales[source_group] = math.sqrt(source_square)
    scales[target_group] = math.sqrt(target_square)
    return GroupDesign(tuple(scales), design.heights)


def calibration_designs(
    baseline: GroupDesign,
    count: int,
) -> tuple[GroupDesign, ...]:
    """生成覆盖六组高度和少量尺度方向的确定性局部标定候选。"""

    if count < 0:
        raise ValueError("标定候选数不能小于 0。")
    candidates: list[GroupDesign] = []
    for direction in (1.0, -1.0):
        for group in range(GROUP_COUNT):
            candidates.append(
                with_height(
                    baseline,
                    group,
                    baseline.heights[group] + direction * 0.25,
                )
            )
    for group in range(GROUP_COUNT):
        candidates.append(
            with_scale(
                baseline,
                group,
                baseline.scales[group] - 0.01,
            )
        )
    return tuple(candidates[:count])


class _SearchContext:
    def __init__(
        self,
        *,
        mother: CampoMotherField,
        coarse_profile: EvaluationProfile,
        reference_profile: EvaluationProfile,
        baseline_design: GroupDesign,
        cache: EvaluationCache,
        target_power_mw: float,
        q_improvement_threshold: float,
        calibration_safety_factor: float,
        calibration_pairs: list[
            tuple[HeterogeneousEvaluation, HeterogeneousEvaluation]
        ],
        baseline_coarse: HeterogeneousEvaluation,
        baseline_reference: HeterogeneousEvaluation,
        progress: ProgressCallback | None,
    ) -> None:
        self.mother = mother
        self.coarse_profile = coarse_profile
        self.reference_profile = reference_profile
        self.cache = cache
        self.target_power_mw = target_power_mw
        self.q_improvement_threshold = q_improvement_threshold
        self.calibration_safety_factor = calibration_safety_factor
        self.calibration_pairs = calibration_pairs
        self.baseline_coarse = baseline_coarse
        self.baseline_reference = baseline_reference
        self.current_design = baseline_design
        self.current_coarse = baseline_coarse
        self.current_reference = baseline_reference
        self.progress = progress
        self.trace: list[SearchStep] = []
        self.calibration = self._rebuild_calibration()

    def _rebuild_calibration(self) -> PowerCalibration:
        return build_power_calibration(
            baseline_coarse=self.baseline_coarse,
            baseline_reference=self.baseline_reference,
            paired_evaluations=self.calibration_pairs,
            safety_factor=self.calibration_safety_factor,
        )

    def _evaluate_coarse(
        self,
        design: GroupDesign,
    ) -> HeterogeneousEvaluation | None:
        try:
            return evaluate_design(
                mother=self.mother,
                design=design,
                profile=self.coarse_profile,
                cache=self.cache,
            )
        except ValueError:
            return None

    def _evaluate_reference(
        self,
        design: GroupDesign,
    ) -> HeterogeneousEvaluation | None:
        try:
            return evaluate_design(
                mother=self.mother,
                design=design,
                profile=self.reference_profile,
                cache=self.cache,
            )
        except ValueError:
            return None

    def try_candidates(
        self,
        *,
        stage: str,
        candidates: Iterable[tuple[str, GroupDesign]],
    ) -> bool:
        ranked: list[
            tuple[float, str, GroupDesign, HeterogeneousEvaluation]
        ] = []
        current_estimated_q = self.calibration.estimated_q_kw_m2(
            self.current_coarse
        )
        for action, design in candidates:
            coarse = self._evaluate_coarse(design)
            if coarse is None:
                continue
            if self.calibration.upper_power_mw(coarse) < self.target_power_mw:
                continue
            estimated_q = self.calibration.estimated_q_kw_m2(coarse)
            if (
                estimated_q
                <= current_estimated_q + self.q_improvement_threshold
            ):
                continue
            ranked.append((estimated_q, action, design, coarse))

        ranked.sort(key=lambda item: item[0], reverse=True)
        for _, action, design, coarse in ranked:
            reference = self._evaluate_reference(design)
            if reference is None:
                continue
            self.calibration_pairs.append((coarse, reference))
            self.calibration = self._rebuild_calibration()
            if not reference.is_feasible(self.target_power_mw):
                continue
            if (
                reference.unit_area_power_kw_m2
                <= self.current_reference.unit_area_power_kw_m2
                + self.q_improvement_threshold
            ):
                continue
            self.current_design = design
            self.current_coarse = coarse
            self.current_reference = reference
            self.trace.append(
                SearchStep(
                    stage=stage,
                    action=action,
                    design=design,
                    evaluation=reference,
                    estimated_power_mw=(
                        self.calibration.estimate_power_mw(coarse)
                    ),
                    empirical_bound_mw=(
                        self.calibration.empirical_bound_mw
                    ),
                )
            )
            if self.progress is not None:
                self.progress(
                    f"{stage} 接受 {action}："
                    f"P={reference.annual_power_mw:.6f} MW，"
                    f"q={reference.unit_area_power_kw_m2:.6f} kW/m²"
                )
            return True
        return False


def _height_candidates(
    design: GroupDesign,
    group: int,
    step: float,
) -> tuple[tuple[str, GroupDesign], ...]:
    return tuple(
        (
            f"H{group + 1}{'+' if direction > 0 else '-'}{step:g}",
            with_height(
                design,
                group,
                design.heights[group] + direction * step,
            ),
        )
        for direction in (-1.0, 1.0)
    )


def _scale_candidates(
    design: GroupDesign,
    group: int,
    step: float,
) -> tuple[tuple[str, GroupDesign], ...]:
    return tuple(
        (
            f"s{group + 1}{'+' if direction > 0 else '-'}{step:g}",
            with_scale(
                design,
                group,
                design.scales[group] + direction * step,
            ),
        )
        for direction in (-1.0, 1.0)
    )


def _transfer_candidates(
    *,
    design: GroupDesign,
    area_fraction: float,
    total_area_m2: float,
    group_counts: tuple[int, ...],
    base_mirror_area_m2: float,
) -> tuple[tuple[str, GroupDesign], ...]:
    preferred_pairs = (
        (3, 2),
        (3, 4),
        (5, 2),
        (5, 4),
    )
    delta_area = total_area_m2 * area_fraction
    candidates: list[tuple[str, GroupDesign]] = []
    for left, right in preferred_pairs:
        for source, target in ((left, right), (right, left)):
            try:
                candidate = transfer_area(
                    design=design,
                    source_group=source,
                    target_group=target,
                    delta_area_m2=delta_area,
                    group_counts=group_counts,
                    base_mirror_area_m2=base_mirror_area_m2,
                )
            except ValueError:
                continue
            candidates.append(
                (
                    f"G{source + 1}->G{target + 1},"
                    f"ΔA={delta_area:.3f}",
                    candidate,
                )
            )
    return tuple(candidates)


def optimize_group_design(
    *,
    mother: CampoMotherField,
    coarse_profile: EvaluationProfile,
    reference_profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    calibration_candidate_count: int = 6,
    calibration_safety_factor: float = 1.2,
    maximum_cycles_per_level: int = 2,
    q_improvement_threshold: float = 1e-5,
    height_steps: tuple[float, ...] = (0.4, 0.2, 0.1),
    scale_steps: tuple[float, ...] = (0.03, 0.015, 0.005),
    area_transfer_fractions: tuple[float, ...] = (0.005, 0.002, 0.001),
    cache: EvaluationCache | None = None,
    progress: ProgressCallback | None = None,
) -> SearchOutcome:
    if maximum_cycles_per_level < 0:
        raise ValueError("maximum_cycles_per_level 不能小于 0。")
    working_cache = cache or EvaluationCache()
    baseline_design = GroupDesign.uniform(
        mother.base_installation_height
    )
    baseline_coarse = evaluate_design(
        mother=mother,
        design=baseline_design,
        profile=coarse_profile,
        cache=working_cache,
    )
    baseline_reference = evaluate_design(
        mother=mother,
        design=baseline_design,
        profile=reference_profile,
        cache=working_cache,
    )

    pairs: list[
        tuple[HeterogeneousEvaluation, HeterogeneousEvaluation]
    ] = []
    for candidate in calibration_designs(
        baseline_design,
        calibration_candidate_count,
    ):
        try:
            coarse = evaluate_design(
                mother=mother,
                design=candidate,
                profile=coarse_profile,
                cache=working_cache,
            )
            reference = evaluate_design(
                mother=mother,
                design=candidate,
                profile=reference_profile,
                cache=working_cache,
            )
        except ValueError:
            continue
        pairs.append((coarse, reference))

    context = _SearchContext(
        mother=mother,
        coarse_profile=coarse_profile,
        reference_profile=reference_profile,
        baseline_design=baseline_design,
        cache=working_cache,
        target_power_mw=target_power_mw,
        q_improvement_threshold=q_improvement_threshold,
        calibration_safety_factor=calibration_safety_factor,
        calibration_pairs=pairs,
        baseline_coarse=baseline_coarse,
        baseline_reference=baseline_reference,
        progress=progress,
    )
    stage_evaluations: list[tuple[str, HeterogeneousEvaluation]] = [
        ("uniform-1471", baseline_reference)
    ]

    for level, step in enumerate(height_steps, start=1):
        for cycle in range(maximum_cycles_per_level):
            improved = False
            order = range(GROUP_COUNT) if cycle % 2 == 0 else reversed(
                range(GROUP_COUNT)
            )
            for group in order:
                improved |= context.try_candidates(
                    stage=f"height-L{level}",
                    candidates=_height_candidates(
                        context.current_design,
                        group,
                        step,
                    ),
                )
            if not improved:
                break
    stage_evaluations.append(("height-only", context.current_reference))

    base_area = (
        mother.mirror_count
        * mother.base_width
        * mother.base_height
    )
    for level, fraction in enumerate(
        area_transfer_fractions,
        start=1,
    ):
        for _ in range(maximum_cycles_per_level):
            improved = context.try_candidates(
                stage=f"transfer-L{level}",
                candidates=_transfer_candidates(
                    design=context.current_design,
                    area_fraction=fraction,
                    total_area_m2=base_area,
                    group_counts=mother.group_counts,
                    base_mirror_area_m2=(
                        mother.base_width * mother.base_height
                    ),
                ),
            )
            if not improved:
                break
    stage_evaluations.append(("height-transfer", context.current_reference))

    for level, step in enumerate(scale_steps, start=1):
        for cycle in range(maximum_cycles_per_level):
            improved = False
            order = range(GROUP_COUNT) if cycle % 2 == 0 else reversed(
                range(GROUP_COUNT)
            )
            for group in order:
                improved |= context.try_candidates(
                    stage=f"scale-L{level}",
                    candidates=_scale_candidates(
                        context.current_design,
                        group,
                        step,
                    ),
                )
            if level <= len(area_transfer_fractions):
                improved |= context.try_candidates(
                    stage=f"rescan-transfer-L{level}",
                    candidates=_transfer_candidates(
                        design=context.current_design,
                        area_fraction=area_transfer_fractions[level - 1],
                        total_area_m2=base_area,
                        group_counts=mother.group_counts,
                        base_mirror_area_m2=(
                            mother.base_width * mother.base_height
                        ),
                    ),
                )
            for group in reversed(range(GROUP_COUNT)):
                improved |= context.try_candidates(
                    stage=f"height-rescan-L{level}",
                    candidates=_height_candidates(
                        context.current_design,
                        group,
                        min(0.1, height_steps[-1]),
                    ),
                )
            if not improved:
                break
    stage_evaluations.append(("height-size", context.current_reference))

    return SearchOutcome(
        baseline_design=baseline_design,
        baseline_evaluation=baseline_reference,
        best_design=context.current_design,
        best_evaluation=context.current_reference,
        calibration=context.calibration,
        calibration_pairs=tuple(context.calibration_pairs),
        trace=tuple(context.trace),
        stage_evaluations=tuple(stage_evaluations),
    )
