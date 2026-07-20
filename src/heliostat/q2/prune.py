"""胜出布局最外层东西对称镜位对的结构化修剪。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    FieldEvaluation,
    LayoutParameters,
    evaluate_coordinates,
)
from .layout import GeneratedLayout


@dataclass(frozen=True)
class PruneStep:
    removed_indices: tuple[int, int]
    evaluation: FieldEvaluation


@dataclass(frozen=True)
class PruneResult:
    initial: FieldEvaluation
    best: FieldEvaluation
    steps: tuple[PruneStep, ...]


def _outer_symmetric_pairs(
    layout: GeneratedLayout,
    ring_count: int,
    *,
    ring_depth: int,
    tolerance: float = 1e-7,
) -> tuple[tuple[int, int], ...]:
    if ring_depth < 1:
        raise ValueError("ring_depth 必须大于等于 1。")
    selected_start = max(0, ring_count - ring_depth)
    offsets: list[int] = []
    running = 0
    for ring in layout.rings[:ring_count]:
        offsets.append(running)
        running += ring.mirror_count

    pairs: list[tuple[int, int]] = []
    for ring_index in range(selected_start, ring_count):
        ring = layout.rings[ring_index]
        offset = offsets[ring_index]
        coordinates = ring.coordinates
        unused = set(range(coordinates.shape[0]))
        while unused:
            local = min(unused)
            unused.remove(local)
            x, y = coordinates[local]
            if abs(float(x)) <= tolerance:
                continue
            candidates = [
                other
                for other in unused
                if abs(float(coordinates[other, 0] + x)) <= tolerance
                and abs(float(coordinates[other, 1] - y)) <= tolerance
            ]
            if not candidates:
                continue
            partner = min(candidates)
            unused.remove(partner)
            pairs.append((offset + local, offset + partner))
    return tuple(pairs)


def prune_outer_symmetric_pairs(
    *,
    layout: GeneratedLayout,
    parameters: LayoutParameters,
    initial: FieldEvaluation,
    profile: EvaluationProfile,
    target_power_mw: float = 42.0,
    ring_depth: int = 2,
    maximum_rounds: int = 10,
    maximum_pairs_per_round: int | None = None,
    cache: EvaluationCache | None = None,
) -> PruneResult:
    """逐轮全场复算，仅接受保持可行且提高单位面积功率的删镜。"""

    if maximum_rounds < 0:
        raise ValueError("maximum_rounds 不能小于 0。")
    if not initial.is_feasible(target_power_mw):
        raise ValueError("结构化修剪要求初始镜场已经满足功率约束。")

    original = layout.prefix(initial.ring_count)
    if original.shape != initial.coordinates.shape or not np.allclose(
        original,
        initial.coordinates,
        atol=1e-9,
    ):
        raise ValueError("initial 坐标与指定布局前缀不一致。")

    pairs = _outer_symmetric_pairs(
        layout,
        initial.ring_count,
        ring_depth=ring_depth,
    )
    active = np.ones(original.shape[0], dtype=bool)
    current = initial
    steps: list[PruneStep] = []

    for _ in range(maximum_rounds):
        remaining_pairs = [
            pair for pair in pairs if active[pair[0]] and active[pair[1]]
        ]
        if (
            maximum_pairs_per_round is not None
            and len(remaining_pairs) > maximum_pairs_per_round
        ):
            sampled_indices = np.linspace(
                0,
                len(remaining_pairs) - 1,
                maximum_pairs_per_round,
                dtype=int,
            )
            remaining_pairs = [remaining_pairs[index] for index in sampled_indices]
        best_pair: tuple[int, int] | None = None
        best_evaluation: FieldEvaluation | None = None

        for pair in remaining_pairs:
            candidate_active = active.copy()
            candidate_active[list(pair)] = False
            candidate = evaluate_coordinates(
                layout_kind=layout.kind,
                ring_count=initial.ring_count,
                coordinates=original[candidate_active],
                parameters=parameters,
                profile=profile,
                cache=cache,
            )
            if not candidate.is_feasible(target_power_mw):
                continue
            if candidate.unit_area_power_kw_m2 <= current.unit_area_power_kw_m2:
                continue
            if (
                best_evaluation is None
                or candidate.unit_area_power_kw_m2
                > best_evaluation.unit_area_power_kw_m2
            ):
                best_pair = pair
                best_evaluation = candidate

        if best_pair is None or best_evaluation is None:
            break
        active[list(best_pair)] = False
        current = best_evaluation
        steps.append(PruneStep(best_pair, current))

    return PruneResult(initial, current, tuple(steps))
