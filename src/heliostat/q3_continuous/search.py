"""独立第三问 Campo 连续规格诊断与分阶段坐标搜索。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    evaluate_design,
)
from .model import CampoMotherField, ContinuousDesign


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CampoDiagnostics:
    radial_rmse_kw_m2: float
    radial_azimuth_rmse_kw_m2: float
    relative_rmse_reduction: float
    azimuth_coefficient_kw_m2: float
    azimuth_recommended: bool


@dataclass(frozen=True)
class SearchStep:
    stage: str
    action: str
    design: ContinuousDesign
    evaluation: HeterogeneousEvaluation
    estimated_power_mw: float


@dataclass(frozen=True)
class SearchOutcome:
    baseline_design: ContinuousDesign
    baseline_evaluation: HeterogeneousEvaluation
    best_design: ContinuousDesign
    best_evaluation: HeterogeneousEvaluation
    diagnostics: CampoDiagnostics
    trace: tuple[SearchStep, ...]
    stage_evaluations: tuple[
        tuple[str, HeterogeneousEvaluation],
        ...,
    ]


def _design_matrix(
    mother: CampoMotherField,
    *,
    include_azimuth: bool,
) -> np.ndarray:
    zone1 = (mother.zone_indices == 1).astype(float)
    zone2 = (mother.zone_indices == 2).astype(float)
    columns = [
        zone1,
        zone2,
        zone1 * mother.normalized_rows,
        zone2 * mother.normalized_rows,
    ]
    if include_azimuth:
        columns.append(mother.azimuth_features)
    return np.column_stack(columns)


def diagnose_campo_structure(
    mother: CampoMotherField,
    evaluation: HeterogeneousEvaluation,
    *,
    recommendation_threshold: float = 0.02,
) -> CampoDiagnostics:
    """比较区域—行号模型与增加同环方位项后的单镜拟合误差。"""

    if evaluation.mirror_count != mother.mirror_count:
        raise ValueError("诊断评价与 Campo 镜场镜子数不一致。")
    mirror_power = np.asarray(
        [
            record.average_output_power_kw
            for record in evaluation.solution.mirror_annual_results
        ],
        dtype=float,
    )
    mirror_area = evaluation.widths * evaluation.heights
    unit_output = mirror_power / mirror_area
    radial = _design_matrix(mother, include_azimuth=False)
    angular = _design_matrix(mother, include_azimuth=True)
    radial_coefficients = np.linalg.lstsq(
        radial,
        unit_output,
        rcond=None,
    )[0]
    angular_coefficients = np.linalg.lstsq(
        angular,
        unit_output,
        rcond=None,
    )[0]
    radial_residual = unit_output - radial @ radial_coefficients
    angular_residual = unit_output - angular @ angular_coefficients
    radial_rmse = float(np.sqrt(np.mean(radial_residual**2)))
    angular_rmse = float(np.sqrt(np.mean(angular_residual**2)))
    reduction = (
        0.0
        if radial_rmse <= 0.0
        else (radial_rmse - angular_rmse) / radial_rmse
    )
    return CampoDiagnostics(
        radial_rmse_kw_m2=radial_rmse,
        radial_azimuth_rmse_kw_m2=angular_rmse,
        relative_rmse_reduction=float(reduction),
        azimuth_coefficient_kw_m2=float(angular_coefficients[-1]),
        azimuth_recommended=bool(reduction >= recommendation_threshold),
    )


def _parameter_candidate(
    design: ContinuousDesign,
    name: str,
    delta: float,
) -> ContinuousDesign:
    return replace(design, **{name: getattr(design, name) + delta})


def _within_search_bounds(
    design: ContinuousDesign,
    *,
    monotone: bool,
) -> bool:
    if monotone and (
        design.size_zone1_slope > 1e-12
        or design.size_zone2_slope > 1e-12
        or design.height_zone1_slope < -1e-12
        or design.height_zone2_slope < -1e-12
    ):
        return False
    if any(
        abs(value) > 0.30
        for value in (
            design.size_zone1_slope,
            design.size_zone2_slope,
            design.size_zone2_offset,
        )
    ):
        return False
    if abs(design.size_azimuth) > 0.15:
        return False
    if any(
        abs(value) > 2.0
        for value in (
            design.height_offset,
            design.height_zone1_slope,
            design.height_zone2_slope,
            design.height_zone2_offset,
        )
    ):
        return False
    if abs(design.height_azimuth) > 1.0:
        return False
    return 0.85 <= design.area_ratio <= 1.05


class _SearchContext:
    def __init__(
        self,
        *,
        mother: CampoMotherField,
        coarse_profile: EvaluationProfile,
        reference_profile: EvaluationProfile,
        current_design: ContinuousDesign,
        current_coarse: HeterogeneousEvaluation,
        current_reference: HeterogeneousEvaluation,
        cache: EvaluationCache,
        target_power_mw: float,
        q_improvement_threshold: float,
        progress: ProgressCallback | None,
        monotone: bool,
    ) -> None:
        self.mother = mother
        self.coarse_profile = coarse_profile
        self.reference_profile = reference_profile
        self.current_design = current_design
        self.current_coarse = current_coarse
        self.current_reference = current_reference
        self.cache = cache
        self.target_power_mw = target_power_mw
        self.q_improvement_threshold = q_improvement_threshold
        self.progress = progress
        self.monotone = monotone
        self.trace: list[SearchStep] = []

    def _evaluate(
        self,
        design: ContinuousDesign,
        profile: EvaluationProfile,
    ) -> HeterogeneousEvaluation | None:
        if not _within_search_bounds(design, monotone=self.monotone):
            return None
        try:
            return evaluate_design(
                mother=self.mother,
                design=design,
                profile=profile,
                cache=self.cache,
            )
        except ValueError:
            return None

    def _estimated_power(
        self,
        coarse: HeterogeneousEvaluation,
    ) -> float:
        return self.current_reference.annual_power_mw + (
            coarse.annual_power_mw
            - self.current_coarse.annual_power_mw
        )

    def _accepts(self, candidate: HeterogeneousEvaluation) -> bool:
        current_feasible = self.current_reference.is_feasible(
            self.target_power_mw
        )
        if not current_feasible:
            return (
                candidate.annual_power_mw
                > self.current_reference.annual_power_mw + 1e-6
            )
        return (
            candidate.is_feasible(self.target_power_mw)
            and candidate.unit_area_power_kw_m2
            > self.current_reference.unit_area_power_kw_m2
            + self.q_improvement_threshold
        )

    def try_candidates(
        self,
        *,
        stage: str,
        candidates: Iterable[tuple[str, ContinuousDesign]],
    ) -> bool:
        ranked: list[
            tuple[
                float,
                str,
                ContinuousDesign,
                HeterogeneousEvaluation,
                float,
            ]
        ] = []
        current_feasible = self.current_reference.is_feasible(
            self.target_power_mw
        )
        for action, design in candidates:
            coarse = self._evaluate(design, self.coarse_profile)
            if coarse is None:
                continue
            estimated_power = self._estimated_power(coarse)
            if estimated_power < self.target_power_mw - 0.35:
                continue
            estimated_q = (
                1000.0 * estimated_power / coarse.total_area_m2
            )
            score = estimated_q if current_feasible else estimated_power
            ranked.append(
                (score, action, design, coarse, estimated_power)
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        for _, action, design, coarse, estimated_power in ranked:
            reference = self._evaluate(design, self.reference_profile)
            if reference is None or not self._accepts(reference):
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
                    estimated_power_mw=estimated_power,
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


def _coordinate_candidates(
    design: ContinuousDesign,
    parameter: str,
    step: float,
) -> tuple[tuple[str, ContinuousDesign], ...]:
    return tuple(
        (
            f"{parameter}{'+' if direction > 0 else '-'}{step:g}",
            _parameter_candidate(
                design,
                parameter,
                direction * step,
            ),
        )
        for direction in (-1.0, 1.0)
    )


def _scan_level(
    context: _SearchContext,
    *,
    stage: str,
    parameters: Sequence[str],
    step: float,
    maximum_cycles: int,
) -> None:
    for cycle in range(maximum_cycles):
        improved = False
        order: Iterable[str] = (
            parameters if cycle % 2 == 0 else reversed(parameters)
        )
        for parameter in order:
            improved |= context.try_candidates(
                stage=stage,
                candidates=_coordinate_candidates(
                    context.current_design,
                    parameter,
                    step,
                ),
            )
        if not improved:
            break


def _build_context(
    *,
    mother: CampoMotherField,
    design: ContinuousDesign,
    coarse_profile: EvaluationProfile,
    reference_profile: EvaluationProfile,
    cache: EvaluationCache,
    target_power_mw: float,
    q_improvement_threshold: float,
    progress: ProgressCallback | None,
    monotone: bool,
) -> _SearchContext:
    coarse = evaluate_design(
        mother=mother,
        design=design,
        profile=coarse_profile,
        cache=cache,
    )
    reference = evaluate_design(
        mother=mother,
        design=design,
        profile=reference_profile,
        cache=cache,
    )
    return _SearchContext(
        mother=mother,
        coarse_profile=coarse_profile,
        reference_profile=reference_profile,
        current_design=design,
        current_coarse=coarse,
        current_reference=reference,
        cache=cache,
        target_power_mw=target_power_mw,
        q_improvement_threshold=q_improvement_threshold,
        progress=progress,
        monotone=monotone,
    )


def optimize_continuous_design(
    *,
    mother: CampoMotherField,
    coarse_profile: EvaluationProfile,
    reference_profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    include_azimuth: bool = False,
    monotone: bool = True,
    maximum_cycles_per_level: int = 2,
    q_improvement_threshold: float = 1e-5,
    height_steps: tuple[float, ...] = (0.4, 0.2, 0.1),
    size_steps: tuple[float, ...] = (0.04, 0.02, 0.01),
    area_steps: tuple[float, ...] = (0.005, 0.002, 0.001),
    cache: EvaluationCache | None = None,
    progress: ProgressCallback | None = None,
) -> SearchOutcome:
    if maximum_cycles_per_level < 0:
        raise ValueError("maximum_cycles_per_level 不能小于 0。")
    working_cache = cache or EvaluationCache()
    baseline_design = ContinuousDesign.uniform()
    context = _build_context(
        mother=mother,
        design=baseline_design,
        coarse_profile=coarse_profile,
        reference_profile=reference_profile,
        cache=working_cache,
        target_power_mw=target_power_mw,
        q_improvement_threshold=q_improvement_threshold,
        progress=progress,
        monotone=monotone,
    )
    baseline_evaluation = context.current_reference
    diagnostics = diagnose_campo_structure(
        mother,
        baseline_evaluation,
    )
    stages: list[tuple[str, HeterogeneousEvaluation]] = [
        ("q2-uniform", baseline_evaluation)
    ]

    height_parameters = [
        "height_offset",
        "height_zone1_slope",
        "height_zone2_slope",
        "height_zone2_offset",
    ]
    if include_azimuth:
        height_parameters.append("height_azimuth")
    for level, step in enumerate(height_steps, start=1):
        _scan_level(
            context,
            stage=f"height-L{level}",
            parameters=height_parameters,
            step=step,
            maximum_cycles=maximum_cycles_per_level,
        )
    stages.append(("height-only", context.current_reference))

    size_parameters = [
        "size_zone1_slope",
        "size_zone2_slope",
        "size_zone2_offset",
    ]
    if include_azimuth:
        size_parameters.append("size_azimuth")
    for level, step in enumerate(size_steps, start=1):
        _scan_level(
            context,
            stage=f"fixed-area-size-L{level}",
            parameters=size_parameters,
            step=step,
            maximum_cycles=maximum_cycles_per_level,
        )
        _scan_level(
            context,
            stage=f"fixed-area-height-rescan-L{level}",
            parameters=height_parameters,
            step=min(step * 5.0, height_steps[-1]),
            maximum_cycles=min(1, maximum_cycles_per_level),
        )
    stages.append(("fixed-area-reallocation", context.current_reference))

    for level, step in enumerate(area_steps, start=1):
        for _ in range(maximum_cycles_per_level):
            improved = context.try_candidates(
                stage=f"area-compression-L{level}",
                candidates=(
                    (
                        f"area_ratio-{step:g}",
                        replace(
                            context.current_design,
                            area_ratio=(
                                context.current_design.area_ratio - step
                            ),
                        ),
                    ),
                ),
            )
            if not improved:
                break
        _scan_level(
            context,
            stage=f"area-size-rescan-L{level}",
            parameters=size_parameters,
            step=size_steps[min(level - 1, len(size_steps) - 1)],
            maximum_cycles=min(1, maximum_cycles_per_level),
        )
        _scan_level(
            context,
            stage=f"area-height-rescan-L{level}",
            parameters=height_parameters,
            step=height_steps[-1],
            maximum_cycles=min(1, maximum_cycles_per_level),
        )
    stages.append(("area-compression", context.current_reference))

    return SearchOutcome(
        baseline_design=baseline_design,
        baseline_evaluation=baseline_evaluation,
        best_design=context.current_design,
        best_evaluation=context.current_reference,
        diagnostics=diagnostics,
        trace=tuple(context.trace),
        stage_evaluations=tuple(stages),
    )


def refine_design_parameters(
    *,
    mother: CampoMotherField,
    initial_design: ContinuousDesign,
    coarse_profile: EvaluationProfile,
    reference_profile: EvaluationProfile,
    parameters: Sequence[str],
    steps: Sequence[float],
    stage: str,
    target_power_mw: float = 42.0,
    monotone: bool = True,
    maximum_cycles_per_level: int = 2,
    q_improvement_threshold: float = 1e-5,
    cache: EvaluationCache | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[ContinuousDesign, HeterogeneousEvaluation, tuple[SearchStep, ...]]:
    working_cache = cache or EvaluationCache()
    context = _build_context(
        mother=mother,
        design=initial_design,
        coarse_profile=coarse_profile,
        reference_profile=reference_profile,
        cache=working_cache,
        target_power_mw=target_power_mw,
        q_improvement_threshold=q_improvement_threshold,
        progress=progress,
        monotone=monotone,
    )
    for level, step in enumerate(steps, start=1):
        _scan_level(
            context,
            stage=f"{stage}-L{level}",
            parameters=parameters,
            step=step,
            maximum_cycles=maximum_cycles_per_level,
        )
    return (
        context.current_design,
        context.current_reference,
        tuple(context.trace),
    )
