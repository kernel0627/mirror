"""统一几何预检、四级精度和六区候选评价。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from ..config import FieldConfig
from ..q2.evaluate import EvaluationProfile
from ._optics import (
    EvaluationCache,
    HeterogeneousEvaluation,
    coarse_profile as _coarse_profile,
    dense_profile as _dense_profile,
    evaluate_specifications as _evaluate_specifications,
    formal_profile as _formal_profile,
    medium_profile as _medium_profile,
    smoke_profile as _smoke_profile,
)
from ._baseline import (
    ExpandedSpecifications,
    HeterogeneousGeometryCheck,
    validate_heterogeneous_field,
)
from .model import RefineBaseline, RefineDesign, RefineField, expand_specifications
from .tower_modes import build_refine_field


@dataclass(frozen=True)
class RefineEvaluation:
    design: RefineDesign
    field: RefineField
    specifications: ExpandedSpecifications
    raw: HeterogeneousEvaluation

    @property
    def profile_name(self) -> str:
        return self.raw.profile_name

    @property
    def mirror_count(self) -> int:
        return self.raw.mirror_count

    @property
    def total_area_m2(self) -> float:
        return self.raw.total_area_m2

    @property
    def annual_power_mw(self) -> float:
        return self.raw.annual_power_mw

    @property
    def unit_area_power_kw_m2(self) -> float:
        return self.raw.unit_area_power_kw_m2

    @property
    def geometry(self) -> HeterogeneousGeometryCheck:
        return self.raw.geometry

    def is_feasible(self, target_power_mw: float = 42.0) -> bool:
        return self.raw.is_feasible(target_power_mw)


def coarse_profile() -> EvaluationProfile:
    return replace(_coarse_profile(), name="q3-six-refine-coarse")


def medium_profile() -> EvaluationProfile:
    return replace(_medium_profile(), name="q3-six-refine-medium")


def formal_profile() -> EvaluationProfile:
    return replace(_formal_profile(), name="q3-six-refine-formal")


def dense_profile(*, neighbor_radius_m: float) -> EvaluationProfile:
    profile = _dense_profile()
    return replace(
        profile,
        name=f"q3-six-refine-dense-{neighbor_radius_m:g}m",
        solver=replace(profile.solver, neighbor_radius_m=neighbor_radius_m),
    )


def smoke_profile() -> EvaluationProfile:
    return replace(_smoke_profile(), name="q3-six-refine-smoke")


def _field_config(baseline: RefineBaseline, design: RefineDesign) -> FieldConfig:
    parameters = baseline.parameters
    return replace(
        FieldConfig(),
        field_radius=parameters.field_radius,
        exclusion_radius=parameters.exclusion_radius,
        tower_x=parameters.tower_x,
        tower_y=design.tower_y,
        mirror_width=parameters.mirror_width,
        mirror_height=parameters.mirror_height,
        mirror_center_z=parameters.installation_height,
    )


def prepare_candidate(
    *,
    baseline: RefineBaseline,
    design: RefineDesign,
) -> tuple[RefineField, ExpandedSpecifications, HeterogeneousGeometryCheck]:
    field = build_refine_field(baseline, design)
    specifications = expand_specifications(field, design)
    check = validate_heterogeneous_field(
        coordinates=field.coordinates,
        widths=specifications.widths,
        heights=specifications.heights,
        installation_heights=specifications.installation_heights,
        tower_x=baseline.parameters.tower_x,
        tower_y=design.tower_y,
        field_radius=baseline.parameters.field_radius,
        exclusion_radius=baseline.parameters.exclusion_radius,
        safety_epsilon=baseline.parameters.safety_epsilon,
    )
    return field, specifications, check


def evaluate_design(
    *,
    baseline: RefineBaseline,
    design: RefineDesign,
    profile: EvaluationProfile,
    cache: EvaluationCache | None = None,
) -> RefineEvaluation:
    field, specifications, check = prepare_candidate(
        baseline=baseline,
        design=design,
    )
    if not check.valid:
        raise ValueError(check.reason or "六区微调候选几何不合法。")
    raw = _evaluate_specifications(
        coordinates=field.coordinates,
        specifications=specifications,
        ring_indices=field.ring_indices,
        group_indices=field.group_indices,
        original_indices=field.original_indices,
        field_config=_field_config(baseline, design),
        profile=profile,
        safety_epsilon=baseline.parameters.safety_epsilon,
        cache=cache,
    )
    return RefineEvaluation(
        design=design,
        field=field,
        specifications=specifications,
        raw=raw,
    )


def metrics(evaluation: RefineEvaluation, *, target_power_mw: float = 42.0) -> dict[str, object]:
    annual = asdict(evaluation.raw.solution.annual_result)
    return {
        "profile": evaluation.profile_name,
        "tower_mode": evaluation.design.tower_mode,
        "tower_x_m": 0.0,
        "tower_y_m": evaluation.design.tower_y,
        "mirror_count": evaluation.mirror_count,
        "mirror_set_hash": evaluation.field.mirror_set_hash,
        "outer_clipped_count": evaluation.field.outer_clipped_count,
        "total_area_m2": evaluation.total_area_m2,
        "annual_power_mw": evaluation.annual_power_mw,
        "power_margin_mw": evaluation.annual_power_mw - target_power_mw,
        "unit_area_power_kw_m2": evaluation.unit_area_power_kw_m2,
        **annual,
    }


__all__ = (
    "EvaluationCache",
    "RefineEvaluation",
    "coarse_profile",
    "dense_profile",
    "evaluate_design",
    "formal_profile",
    "medium_profile",
    "metrics",
    "prepare_candidate",
    "smoke_profile",
)
