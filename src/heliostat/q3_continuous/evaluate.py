"""独立第三问连续异构镜场评价、缓存和多级精度配置。"""

from __future__ import annotations

import hashlib
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
    ContinuousDesign,
    ExpandedSpecifications,
    HeterogeneousGeometryCheck,
    expand_continuous_design,
    validate_heterogeneous_field,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HeterogeneousEvaluation:
    """一套逐镜规格与 Campo 结构标签的完整评价结果。"""

    profile_name: str
    coordinates: FloatArray
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    ring_indices: IntArray
    zone_indices: IntArray
    zone_row_indices: IntArray
    normalized_rows: FloatArray
    azimuth_angles: FloatArray
    azimuth_features: FloatArray
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
    """按镜位、逐镜规格、塔位和数值精度缓存评价。"""

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
    coordinates: FloatArray,
    specifications: ExpandedSpecifications,
    ring_indices: IntArray,
    zone_indices: IntArray,
    zone_row_indices: IntArray,
    normalized_rows: FloatArray,
    azimuth_angles: FloatArray,
    azimuth_features: FloatArray,
    nominal_ring_counts: IntArray,
    actual_ring_counts: IntArray,
    original_indices: IntArray,
    field_config: FieldConfig,
    profile: EvaluationProfile,
    safety_epsilon: float = 0.01,
    cache: EvaluationCache | None = None,
) -> HeterogeneousEvaluation:
    xy = np.asarray(coordinates, dtype=float)
    count = int(xy.shape[0])
    structural = {
        "ring_indices": np.asarray(ring_indices, dtype=np.int64),
        "zone_indices": np.asarray(zone_indices, dtype=np.int64),
        "zone_row_indices": np.asarray(
            zone_row_indices,
            dtype=np.int64,
        ),
        "normalized_rows": np.asarray(normalized_rows, dtype=float),
        "azimuth_angles": np.asarray(azimuth_angles, dtype=float),
        "azimuth_features": np.asarray(azimuth_features, dtype=float),
        "nominal_ring_counts": np.asarray(
            nominal_ring_counts,
            dtype=np.int64,
        ),
        "actual_ring_counts": np.asarray(
            actual_ring_counts,
            dtype=np.int64,
        ),
        "original_indices": np.asarray(original_indices, dtype=np.int64),
    }
    for name, values in structural.items():
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
        solution=solution,
        geometry=geometry,
        **structural,
    )


def evaluate_design(
    *,
    mother: CampoMotherField,
    design: ContinuousDesign,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> HeterogeneousEvaluation:
    specifications = expand_continuous_design(mother, design)
    return evaluate_specifications(
        coordinates=mother.coordinates,
        specifications=specifications,
        ring_indices=mother.ring_indices,
        zone_indices=mother.zone_indices,
        zone_row_indices=mother.zone_row_indices,
        normalized_rows=mother.normalized_rows,
        azimuth_angles=mother.azimuth_angles,
        azimuth_features=mother.azimuth_features,
        nominal_ring_counts=mother.nominal_ring_counts,
        actual_ring_counts=mother.actual_ring_counts,
        original_indices=mother.original_indices,
        field_config=field_config_from_mother(mother),
        profile=profile,
        safety_epsilon=mother.parameters.safety_epsilon,
        cache=cache,
    )
