"""径向—角度连续 Campo 的完整异构光学评价与精度配置。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from ..config import FieldConfig, SolverConfig
from ..geometry import prepare_field
from ..q1.aggregate import Question1Solution
from ..q1.solve import SOLAR_TIMES, solve_question1
from ..q2.evaluate import EvaluationProfile
from .model import (
    Campo2DBase,
    Campo2DDesign,
    Campo2DField,
    ExpandedSpecifications,
    HeterogeneousGeometryCheck,
    build_campo_field,
    expand_design,
    validate_heterogeneous_field,
)


@dataclass(frozen=True)
class Campo2DEvaluation:
    profile_name: str
    design: Campo2DDesign
    field: Campo2DField
    specifications: ExpandedSpecifications
    solution: Question1Solution
    geometry: HeterogeneousGeometryCheck

    @property
    def mirror_count(self) -> int:
        return self.field.mirror_count

    @property
    def total_area_m2(self) -> float:
        return self.specifications.total_area_m2

    @property
    def annual_power_mw(self) -> float:
        return self.solution.annual_result.field_output_mw

    @property
    def unit_area_power_kw_m2(self) -> float:
        return self.solution.annual_result.unit_area_output_kw_m2

    def is_feasible(self, target_power_mw: float = 42.0) -> bool:
        return self.annual_power_mw >= target_power_mw


class EvaluationCache:
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
        name="q3-campo2d-coarse",
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
        name="q3-campo2d-medium",
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
        name="q3-campo2d-formal",
        solver=SolverConfig(
            shadow_grid_size=15,
            truncation_rays=256,
            neighbor_radius_m=60.0,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def dense_profile(*, neighbor_radius_m: float = 80.0) -> EvaluationProfile:
    return EvaluationProfile(
        name=f"q3-campo2d-dense-{neighbor_radius_m:g}m",
        solver=SolverConfig(
            shadow_grid_size=20,
            truncation_rays=512,
            neighbor_radius_m=neighbor_radius_m,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def smoke_profile() -> EvaluationProfile:
    return EvaluationProfile(
        name="q3-campo2d-smoke",
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


def field_config(field: Campo2DField) -> FieldConfig:
    return replace(
        FieldConfig(),
        field_radius=field.parameters.field_radius,
        exclusion_radius=field.parameters.exclusion_radius,
        tower_x=field.parameters.tower_x,
        tower_y=field.parameters.tower_y,
        mirror_width=field.base_width,
        mirror_height=field.base_height,
        mirror_center_z=field.base_installation_height,
    )


def _cache_key(
    *,
    field: Campo2DField,
    specifications: ExpandedSpecifications,
    config: FieldConfig,
    profile: EvaluationProfile,
) -> str:
    digest = hashlib.sha256()
    for values in (
        field.coordinates,
        specifications.widths,
        specifications.heights,
        specifications.installation_heights,
    ):
        rounded = np.round(np.asarray(values, dtype="<f8"), decimals=9)
        digest.update(rounded.tobytes(order="C"))
    digest.update(repr(config.to_dict()).encode("utf-8"))
    digest.update(repr(profile.solver.to_dict()).encode("utf-8"))
    digest.update(repr(profile.months).encode("ascii"))
    digest.update(repr(profile.solar_times).encode("ascii"))
    return digest.hexdigest()


def evaluate_field(
    *,
    design: Campo2DDesign,
    field: Campo2DField,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> Campo2DEvaluation:
    specifications = expand_design(field, design)
    geometry = validate_heterogeneous_field(
        field=field,
        specifications=specifications,
    )
    if not geometry.valid:
        raise ValueError(geometry.reason or "Campo2D 候选不满足几何约束。")
    config = field_config(field)
    key = _cache_key(
        field=field,
        specifications=specifications,
        config=config,
        profile=profile,
    )
    solution = cache.get(key) if cache is not None else None
    if solution is None:
        prepared = prepare_field(
            field.coordinates,
            config,
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
    return Campo2DEvaluation(
        profile_name=profile.name,
        design=design.canonical(),
        field=field,
        specifications=specifications,
        solution=solution,
        geometry=geometry,
    )


def evaluate_design(
    *,
    base: Campo2DBase,
    design: Campo2DDesign,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> Campo2DEvaluation:
    field = build_campo_field(base, design)
    return evaluate_field(
        design=design,
        field=field,
        profile=profile,
        cache=cache,
    )
