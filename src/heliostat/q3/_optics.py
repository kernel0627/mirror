"""第三问异构镜场评价、缓存和精度配置。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import FieldConfig, SolverConfig
from ..geometry import prepare_field
from ..q1.aggregate import Question1Solution
from ..q1.solve import SOLAR_TIMES, solve_question1
from ..q2.evaluate import EvaluationProfile
from ._baseline import (
    ExpandedSpecifications,
    HeterogeneousGeometryCheck,
    validate_heterogeneous_field,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HeterogeneousEvaluation:
    """一套确定逐镜规格和活跃镜集合的完整评价结果。"""

    profile_name: str
    coordinates: FloatArray
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    ring_indices: IntArray
    group_indices: IntArray
    original_indices: IntArray
    solution: Question1Solution
    geometry: HeterogeneousGeometryCheck

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.widths * self.heights))

    @property
    def annual_power_mw(self) -> float:
        return self.solution.annual_result.field_output_mw

    @property
    def unit_area_power_kw_m2(self) -> float:
        return self.solution.annual_result.unit_area_output_kw_m2

    def is_feasible(self, target_power_mw: float = 42.0) -> bool:
        return self.annual_power_mw >= target_power_mw


class EvaluationCache:
    """按坐标、逐镜规格、塔位和数值精度缓存第三问评价。"""

    def __init__(self) -> None:
        self._values: dict[str, Question1Solution] = {}

    def get(self, key: str) -> Question1Solution | None:
        return self._values.get(key)

    def put(self, key: str, value: Question1Solution) -> None:
        self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)


def coarse_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-coarse",
        solver=SolverConfig(
            shadow_grid_size=5,
            truncation_rays=64,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
        months=(3, 6, 9, 12),
        solar_times=SOLAR_TIMES,
    )


def medium_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-medium",
        solver=SolverConfig(
            shadow_grid_size=10,
            truncation_rays=128,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def formal_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-formal",
        solver=SolverConfig(
            shadow_grid_size=15,
            truncation_rays=256,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def dense_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-dense",
        solver=SolverConfig(
            shadow_grid_size=20,
            truncation_rays=512,
            neighbor_radius_m=80.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def smoke_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-smoke",
        solver=SolverConfig(
            shadow_grid_size=2,
            truncation_rays=4,
            neighbor_radius_m=60.0,
            truncation_chunk_size=64,
            sobol_seed=2023,
        ),
        months=(6,),
        solar_times=(12.0,),
    )


def _cache_key(
    *,
    coordinates: FloatArray,
    specifications: ExpandedSpecifications,
    field_config: FieldConfig,
    profile: EvaluationProfile,
) -> str:
    digest = hashlib.sha256()
    for values in (
        coordinates,
        specifications.widths,
        specifications.heights,
        specifications.installation_heights,
    ):
        rounded = np.round(np.asarray(values, dtype="<f8"), decimals=9)
        digest.update(rounded.tobytes(order="C"))
    digest.update(repr(field_config.to_dict()).encode("utf-8"))
    digest.update(repr(profile.solver.to_dict()).encode("utf-8"))
    digest.update(repr(profile.months).encode("ascii"))
    digest.update(repr(profile.solar_times).encode("ascii"))
    return digest.hexdigest()


def evaluate_specifications(
    *,
    coordinates: FloatArray,
    specifications: ExpandedSpecifications,
    ring_indices: IntArray,
    group_indices: IntArray,
    original_indices: IntArray,
    field_config: FieldConfig,
    profile: EvaluationProfile,
    safety_epsilon: float = 0.01,
    cache: EvaluationCache | None = None,
) -> HeterogeneousEvaluation:
    xy = np.asarray(coordinates, dtype=float)
    rings = np.asarray(ring_indices, dtype=np.int64)
    groups = np.asarray(group_indices, dtype=np.int64)
    originals = np.asarray(original_indices, dtype=np.int64)
    count = int(xy.shape[0])
    for name, values in (
        ("ring_indices", rings),
        ("group_indices", groups),
        ("original_indices", originals),
    ):
        if values.ndim != 1 or values.shape[0] != count:
            raise ValueError(f"{name} 长度与镜子数不一致。")

    geometry = validate_heterogeneous_field(
        coordinates=xy,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        tower_x=field_config.tower_x,
        tower_y=field_config.tower_y,
        field_radius=field_config.field_radius,
        exclusion_radius=field_config.exclusion_radius,
        safety_epsilon=safety_epsilon,
    )
    if not geometry.valid:
        raise ValueError(geometry.reason or "异构镜场几何约束不合法。")

    key = _cache_key(
        coordinates=xy,
        specifications=specifications,
        field_config=field_config,
        profile=profile,
    )
    solution = cache.get(key) if cache is not None else None
    if solution is None:
        prepared = prepare_field(
            xy,
            field_config,
            mirror_widths=specifications.widths,
            mirror_heights=specifications.heights,
            mirror_center_zs=specifications.installation_heights,
        )
        solution = solve_question1(
            prepared=prepared,
            solver=profile.solver,
            months=profile.months,
            solar_times=profile.solar_times,
        )
        if cache is not None:
            cache.put(key, solution)

    return HeterogeneousEvaluation(
        profile_name=profile.name,
        coordinates=xy,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        ring_indices=rings,
        group_indices=groups,
        original_indices=originals,
        solution=solution,
        geometry=geometry,
    )

