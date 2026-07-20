"""第二问候选镜场的光学评价、缓存和外边界扫描。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..config import FieldConfig, SolverConfig
from ..geometry import prepare_field
from ..q1.aggregate import Question1Solution
from ..q1.solve import SOLAR_TIMES, solve_question1
from .layout import (
    CampoParameters,
    CommonParameters,
    GeneratedLayout,
    PartitionedRingParameters,
)


LayoutParameters = PartitionedRingParameters | CampoParameters


@dataclass(frozen=True)
class EvaluationProfile:
    """同一物理模型下的一组数值离散精度。"""

    name: str
    solver: SolverConfig
    months: tuple[int, ...] = tuple(range(1, 13))
    solar_times: tuple[float, ...] = SOLAR_TIMES


@dataclass(frozen=True)
class FieldEvaluation:
    """一个确定镜场外边界下的完整评价结果。"""

    layout_kind: str
    ring_count: int
    mirror_count: int
    mirror_area_m2: float
    total_area_m2: float
    coordinates: np.ndarray
    solution: Question1Solution

    @property
    def annual_power_mw(self) -> float:
        return self.solution.annual_result.field_output_mw

    @property
    def unit_area_power_kw_m2(self) -> float:
        return self.solution.annual_result.unit_area_output_kw_m2

    def is_feasible(self, target_power_mw: float = 42.0) -> bool:
        return self.annual_power_mw >= target_power_mw


@dataclass(frozen=True)
class ExtentScanResult:
    """一组布局参数在若干镜场外边界中的最好结果。"""

    best: FieldEvaluation
    evaluations: tuple[FieldEvaluation, ...]
    first_feasible_ring_count: int | None


class EvaluationCache:
    """按坐标、塔和数值精度缓存昂贵的光学评价。"""

    def __init__(self) -> None:
        self._values: dict[str, Question1Solution] = {}

    def get(self, key: str) -> Question1Solution | None:
        return self._values.get(key)

    def put(self, key: str, value: Question1Solution) -> None:
        self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)


def exploration_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="exploration",
        solver=SolverConfig(
            shadow_grid_size=5,
            truncation_rays=64,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def refinement_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="refinement",
        solver=SolverConfig(
            shadow_grid_size=10,
            truncation_rays=128,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def final_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="final",
        solver=SolverConfig(
            shadow_grid_size=15,
            truncation_rays=256,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def _field_config(
    parameters: CommonParameters,
    base: FieldConfig | None = None,
) -> FieldConfig:
    config = base or FieldConfig()
    return replace(
        config,
        field_radius=parameters.field_radius,
        exclusion_radius=parameters.exclusion_radius,
        tower_x=parameters.tower_x,
        tower_y=parameters.tower_y,
        mirror_width=parameters.mirror_width,
        mirror_height=parameters.mirror_height,
        mirror_center_z=parameters.installation_height,
    )


def _cache_key(
    coordinates: np.ndarray,
    config: FieldConfig,
    profile: EvaluationProfile,
) -> str:
    digest = hashlib.sha256()
    rounded = np.round(np.asarray(coordinates, dtype="<f8"), decimals=9)
    digest.update(rounded.tobytes(order="C"))
    digest.update(repr(config.to_dict()).encode("utf-8"))
    digest.update(repr(profile.solver.to_dict()).encode("utf-8"))
    digest.update(repr(profile.months).encode("ascii"))
    digest.update(repr(profile.solar_times).encode("ascii"))
    return digest.hexdigest()


def evaluate_coordinates(
    *,
    layout_kind: str,
    ring_count: int,
    coordinates: np.ndarray,
    parameters: LayoutParameters,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
    base_field_config: FieldConfig | None = None,
) -> FieldEvaluation:
    """直接复用问题一模型评价一套确定坐标。"""

    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("候选镜场坐标必须为非空 N×2 数组。")

    config = _field_config(parameters, base_field_config)
    key = _cache_key(xy, config, profile)
    solution = cache.get(key) if cache is not None else None
    if solution is None:
        prepared = prepare_field(xy, config)
        solution = solve_question1(
            prepared=prepared,
            solver=profile.solver,
            months=profile.months,
            solar_times=profile.solar_times,
        )
        if cache is not None:
            cache.put(key, solution)

    mirror_area = parameters.mirror_width * parameters.mirror_height
    return FieldEvaluation(
        layout_kind=layout_kind,
        ring_count=ring_count,
        mirror_count=int(xy.shape[0]),
        mirror_area_m2=mirror_area,
        total_area_m2=float(xy.shape[0] * mirror_area),
        coordinates=xy,
        solution=solution,
    )


def better_evaluation(
    left: FieldEvaluation,
    right: FieldEvaluation,
    *,
    target_power_mw: float = 42.0,
) -> FieldEvaluation:
    """按可行性优先规则返回较优结果。"""

    left_feasible = left.is_feasible(target_power_mw)
    right_feasible = right.is_feasible(target_power_mw)
    if left_feasible != right_feasible:
        return left if left_feasible else right
    if left_feasible:
        if left.unit_area_power_kw_m2 != right.unit_area_power_kw_m2:
            return (
                left
                if left.unit_area_power_kw_m2 > right.unit_area_power_kw_m2
                else right
            )
        return left if left.annual_power_mw <= right.annual_power_mw else right
    return left if left.annual_power_mw >= right.annual_power_mw else right


def _unique_ring_counts(values: Sequence[int], total: int) -> tuple[int, ...]:
    return tuple(sorted({value for value in values if 1 <= value <= total}))


def scan_layout_extents(
    layout: GeneratedLayout,
    parameters: LayoutParameters,
    profile: EvaluationProfile,
    *,
    target_power_mw: float = 42.0,
    coarse_stride: int = 4,
    window: int = 2,
    cache: EvaluationCache | None = None,
    base_field_config: FieldConfig | None = None,
) -> ExtentScanResult:
    """先粗定位功率阈值，再评价阈值附近的连续圆环外边界。"""

    if not layout.rings:
        raise ValueError("布局中没有可用于评价的圆环。")
    if coarse_stride < 1:
        raise ValueError("coarse_stride 必须大于等于 1。")
    if window < 0:
        raise ValueError("window 不能小于 0。")

    total_rings = len(layout.rings)
    coarse_counts = list(range(coarse_stride, total_rings + 1, coarse_stride))
    if not coarse_counts or coarse_counts[-1] != total_rings:
        coarse_counts.append(total_rings)

    evaluated: dict[int, FieldEvaluation] = {}

    def evaluate(ring_count: int) -> FieldEvaluation:
        previous = evaluated.get(ring_count)
        if previous is not None:
            return previous
        value = evaluate_coordinates(
            layout_kind=layout.kind,
            ring_count=ring_count,
            coordinates=layout.prefix(ring_count),
            parameters=parameters,
            profile=profile,
            cache=cache,
            base_field_config=base_field_config,
        )
        evaluated[ring_count] = value
        return value

    first_feasible: int | None = None
    previous_coarse = 0
    for ring_count in coarse_counts:
        value = evaluate(ring_count)
        if value.is_feasible(target_power_mw):
            for refined_count in range(previous_coarse + 1, ring_count + 1):
                refined = evaluate(refined_count)
                if refined.is_feasible(target_power_mw):
                    first_feasible = refined_count
                    break
            break
        previous_coarse = ring_count

    center = first_feasible if first_feasible is not None else total_rings
    local_counts = _unique_ring_counts(
        range(center - window, center + window + 1),
        total_rings,
    )
    for ring_count in local_counts:
        evaluate(ring_count)

    values = tuple(evaluated[key] for key in sorted(evaluated))
    best = values[0]
    for value in values[1:]:
        best = better_evaluation(
            best,
            value,
            target_power_mw=target_power_mw,
        )
    return ExtentScanResult(
        best=best,
        evaluations=values,
        first_feasible_ring_count=first_feasible,
    )
