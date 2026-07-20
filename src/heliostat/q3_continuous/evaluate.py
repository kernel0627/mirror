"""五节点径向连续异构镜场的完整光学评价与精度配置。"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..config import FieldConfig, SolverConfig
from ..geometry import prepare_field
from ..q1.aggregate import Question1Solution
from ..q1.solve import SOLAR_TIMES, solve_question1
from ..q2.evaluate import EvaluationProfile
from .model import (
    CampoMotherField,
    ExpandedSpecifications,
    HeterogeneousGeometryCheck,
    SplineDesign,
    expand_spline_design,
    validate_heterogeneous_field,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HeterogeneousEvaluation:
    profile_name: str
    coordinates: FloatArray
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    ring_indices: IntArray
    ring_radii: FloatArray
    zone_indices: IntArray
    nominal_ring_counts: IntArray
    actual_ring_counts: IntArray
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
    """线程安全的逐规格完整评价缓存。"""

    def __init__(self) -> None:
        self._values: dict[str, Question1Solution] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Question1Solution | None:
        with self._lock:
            return self._values.get(key)

    def put(self, key: str, value: Question1Solution) -> None:
        with self._lock:
            self._values[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


def medium_profile() -> EvaluationProfile:
    """所有参数接受使用的 60 状态中精度。"""

    return EvaluationProfile(
        name="q3-continuous-medium",
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
        name="q3-continuous-formal",
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
        name="q3-continuous-dense",
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
        name="q3-continuous-smoke",
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


def field_config_from_mother(mother: CampoMotherField) -> FieldConfig:
    return replace(
        FieldConfig(),
        field_radius=mother.parameters.field_radius,
        exclusion_radius=mother.parameters.exclusion_radius,
        tower_x=mother.parameters.tower_x,
        tower_y=mother.parameters.tower_y,
        mirror_width=mother.base_width,
        mirror_height=mother.base_height,
        mirror_center_z=mother.base_installation_height,
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
    mother: CampoMotherField,
    specifications: ExpandedSpecifications,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> HeterogeneousEvaluation:
    geometry = validate_heterogeneous_field(
        coordinates=mother.coordinates,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        tower_x=mother.parameters.tower_x,
        tower_y=mother.parameters.tower_y,
        field_radius=mother.parameters.field_radius,
        exclusion_radius=mother.parameters.exclusion_radius,
        safety_epsilon=mother.parameters.safety_epsilon,
    )
    if not geometry.valid:
        raise ValueError(geometry.reason or "异构镜场几何约束不合法。")

    field_config = field_config_from_mother(mother)
    key = _cache_key(
        coordinates=mother.coordinates,
        specifications=specifications,
        field_config=field_config,
        profile=profile,
    )
    solution = cache.get(key) if cache is not None else None
    if solution is None:
        prepared = prepare_field(
            mother.coordinates,
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
        coordinates=mother.coordinates,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        ring_indices=mother.ring_indices,
        ring_radii=mother.ring_radii,
        zone_indices=mother.zone_indices,
        nominal_ring_counts=mother.nominal_ring_counts,
        actual_ring_counts=mother.actual_ring_counts,
        original_indices=mother.original_indices,
        solution=solution,
        geometry=geometry,
    )


def evaluate_design(
    *,
    mother: CampoMotherField,
    design: SplineDesign,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> HeterogeneousEvaluation:
    return evaluate_specifications(
        mother=mother,
        specifications=expand_spline_design(mother, design),
        profile=profile,
        cache=cache,
    )
