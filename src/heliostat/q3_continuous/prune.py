"""独立第三问 Campo 外边界低贡献东西对称镜位复算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    evaluate_specifications,
    field_config_from_mother,
)
from .model import CampoMotherField, ExpandedSpecifications


@dataclass(frozen=True)
class PruneStep:
    removed_original_indices: tuple[int, ...]
    evaluation: HeterogeneousEvaluation


@dataclass(frozen=True)
class PruneOutcome:
    initial: HeterogeneousEvaluation
    best: HeterogeneousEvaluation
    steps: tuple[PruneStep, ...]


def symmetric_pairs(
    evaluation: HeterogeneousEvaluation,
    *,
    tolerance: float = 1e-7,
) -> tuple[tuple[int, int], ...]:
    coordinates = evaluation.coordinates
    unused = set(range(evaluation.mirror_count))
    pairs: list[tuple[int, int]] = []
    while unused:
        index = min(unused)
        unused.remove(index)
        x_m, y_m = coordinates[index]
        if abs(float(x_m)) <= tolerance:
            continue
        partners = [
            other
            for other in unused
            if abs(float(coordinates[other, 0] + x_m)) <= tolerance
            and abs(float(coordinates[other, 1] - y_m)) <= tolerance
        ]
        if not partners:
            continue
        partner = min(partners)
        unused.remove(partner)
        pairs.append((index, partner))
    return tuple(pairs)


def _rank_pairs(
    evaluation: HeterogeneousEvaluation,
    pairs: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    mirror_power = np.array(
        [
            record.average_output_power_kw
            for record in evaluation.solution.mirror_annual_results
        ],
        dtype=float,
    )

    def key(pair: tuple[int, int]) -> tuple[int, float, int]:
        zones = evaluation.zone_indices[list(pair)]
        rings = evaluation.ring_indices[list(pair)]
        preferred = bool(
            np.any(zones == 2)
            and np.any(rings >= 24)
        )
        return (
            0 if preferred else 1,
            float(np.sum(mirror_power[list(pair)])),
            -int(np.max(rings)),
        )

    return sorted(pairs, key=key)


def _remove_pair(
    *,
    current: HeterogeneousEvaluation,
    pair: tuple[int, int],
    mother: CampoMotherField,
    profile: EvaluationProfile,
    cache: EvaluationCache | None,
) -> HeterogeneousEvaluation:
    active = np.ones(current.mirror_count, dtype=bool)
    active[list(pair)] = False
    specifications = ExpandedSpecifications(
        widths=current.widths[active],
        heights=current.heights[active],
        installation_heights=current.installation_heights[active],
        areas=current.widths[active] * current.heights[active],
        scales=current.widths[active] / mother.base_width,
        size_shape=np.log(current.widths[active] / mother.base_width),
        area_normalizer=1.0,
    )
    active_rings = current.ring_indices[active]
    active_angles = current.azimuth_angles[active]
    active_features = np.empty(active_angles.shape[0], dtype=float)
    active_counts = np.empty(active_angles.shape[0], dtype=np.int64)
    cosines = np.cos(active_angles)
    for ring in np.unique(active_rings):
        in_ring = active_rings == ring
        count = int(np.count_nonzero(in_ring))
        active_features[in_ring] = (
            cosines[in_ring] - float(np.mean(cosines[in_ring]))
        )
        active_counts[in_ring] = count
    return evaluate_specifications(
        coordinates=current.coordinates[active],
        specifications=specifications,
        ring_indices=active_rings,
        zone_indices=current.zone_indices[active],
        zone_row_indices=current.zone_row_indices[active],
        normalized_rows=current.normalized_rows[active],
        azimuth_angles=active_angles,
        azimuth_features=active_features,
        nominal_ring_counts=current.nominal_ring_counts[active],
        actual_ring_counts=active_counts,
        original_indices=current.original_indices[active],
        field_config=field_config_from_mother(mother),
        profile=profile,
        safety_epsilon=mother.parameters.safety_epsilon,
        cache=cache,
    )


def prune_symmetric_pairs(
    *,
    mother: CampoMotherField,
    initial: HeterogeneousEvaluation,
    profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    maximum_rounds: int = 4,
    maximum_pairs_per_round: int = 12,
    q_improvement_threshold: float = 1e-5,
    cache: EvaluationCache | None = None,
) -> PruneOutcome:
    if maximum_rounds < 0:
        raise ValueError("maximum_rounds 不能小于 0。")
    if maximum_pairs_per_round < 1:
        raise ValueError("maximum_pairs_per_round 必须大于等于 1。")
    if not initial.is_feasible(target_power_mw):
        raise ValueError("结构化删镜要求初始方案满足功率约束。")

    current = initial
    steps: list[PruneStep] = []
    for _ in range(maximum_rounds):
        pairs = _rank_pairs(current, symmetric_pairs(current))
        pairs = pairs[:maximum_pairs_per_round]
        best_pair: tuple[int, int] | None = None
        best_candidate: HeterogeneousEvaluation | None = None
        for pair in pairs:
            candidate = _remove_pair(
                current=current,
                pair=pair,
                mother=mother,
                profile=profile,
                cache=cache,
            )
            if not candidate.is_feasible(target_power_mw):
                continue
            if (
                candidate.unit_area_power_kw_m2
                <= current.unit_area_power_kw_m2
                + q_improvement_threshold
            ):
                continue
            if (
                best_candidate is None
                or candidate.unit_area_power_kw_m2
                > best_candidate.unit_area_power_kw_m2
            ):
                best_pair = pair
                best_candidate = candidate
        if best_pair is None or best_candidate is None:
            break
        removed = tuple(
            int(current.original_indices[index])
            for index in best_pair
        )
        current = best_candidate
        steps.append(PruneStep(removed, current))

    return PruneOutcome(
        initial=initial,
        best=current,
        steps=tuple(steps),
    )
