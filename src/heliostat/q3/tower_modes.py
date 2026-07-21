"""塔位模式 A/B 和动态 Campo 前缀构造。"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ..q2.layout import GeneratedLayout, generate_campo_layout
from ._baseline import GROUP_RING_RANGES
from .model import RefineBaseline, RefineDesign, RefineField


RING_COUNT = 28


def _group_for_ring(ring_index: int) -> int:
    for group, (start, stop) in enumerate(GROUP_RING_RANGES):
        if start <= ring_index <= stop:
            return group
    raise ValueError(f"圆环 {ring_index} 不属于六区。")


def _membership_hash(
    *,
    layout: GeneratedLayout,
    geometry_center_y: float,
) -> str:
    digest = hashlib.sha256()
    for ring_index, ring in enumerate(layout.rings, start=1):
        angles = np.mod(
            np.arctan2(
                ring.coordinates[:, 0],
                ring.coordinates[:, 1] - geometry_center_y,
            ),
            2.0 * np.pi,
        )
        digest.update(np.asarray((ring_index, ring.nominal_count), dtype="<i8").tobytes())
        digest.update(np.round(angles, 10).astype("<f8").tobytes())
    return digest.hexdigest()


def build_refine_field(
    baseline: RefineBaseline,
    design: RefineDesign,
) -> RefineField:
    """按单一塔位语义重建 Campo，不在轨迹内切换语义。"""

    geometry_center_y = (
        design.tower_y
        if design.tower_mode == "A"
        else baseline.design.tower_y
    )
    parameters = replace(
        baseline.parameters,
        tower_y=geometry_center_y,
        initial_spacing=design.initial_spacing,
        spacing_growth=design.spacing_growth,
    )
    generated = generate_campo_layout(parameters)
    if len(generated.rings) < RING_COUNT:
        raise ValueError(
            f"候选只生成 {len(generated.rings)} 个有效环，不能保留前 28 环。"
        )
    layout = GeneratedLayout("campo", generated.rings[:RING_COUNT])
    coordinates: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    originals: list[np.ndarray] = []
    cursor = 0
    clipped = 0
    for display_index, ring in enumerate(layout.rings, start=1):
        count = ring.mirror_count
        coordinates.append(np.asarray(ring.coordinates, dtype=float))
        rings.append(np.full(count, display_index, dtype=np.int64))
        groups.append(np.full(count, _group_for_ring(display_index), dtype=np.int64))
        originals.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
        clipped += ring.nominal_count - count
    return RefineField(
        coordinates=np.concatenate(coordinates),
        ring_indices=np.concatenate(rings),
        group_indices=np.concatenate(groups),
        original_indices=np.concatenate(originals),
        mirror_set_hash=_membership_hash(
            layout=layout,
            geometry_center_y=geometry_center_y,
        ),
        outer_clipped_count=int(clipped),
        geometry_center_y=float(geometry_center_y),
    )
