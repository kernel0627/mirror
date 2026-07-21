"""六区规格敏感性筛选与径向边界局部扰动。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .model import RefineDesign, RefineField


SPECIFICATION_VARIABLES = tuple(
    f"{prefix}{group}"
    for prefix in ("w", "h", "H")
    for group in range(1, 7)
)
BASE_BOUNDARIES = (1, 5, 11, 14, 20)
BOUNDARY_SHIFTS = (-2, -1, 1, 2)
RING_COUNT = 28


@dataclass(frozen=True)
class Perturbation:
    parameter: str
    group_id: int
    direction: str
    old_value: float
    new_value: float
    design: RefineDesign


@dataclass(frozen=True)
class BoundaryPerturbation:
    """只移动一条内部边界的候选。"""

    boundary_id: int
    shift_rings: int
    boundaries: tuple[int, ...]

    @property
    def original_end_ring(self) -> int:
        return BASE_BOUNDARIES[self.boundary_id - 1]

    @property
    def new_end_ring(self) -> int:
        return self.boundaries[self.boundary_id - 1]

    @property
    def label(self) -> str:
        sign = "+" if self.shift_rings > 0 else ""
        return f"B{self.boundary_id}{sign}{self.shift_rings}"


def validate_boundaries(boundaries: tuple[int, ...]) -> None:
    """验证五条边界能形成六个非空、连续的径向分区。"""

    if len(boundaries) != len(BASE_BOUNDARIES):
        raise ValueError("六区划分必须包含五条内部边界。")
    if not all(isinstance(value, int) for value in boundaries):
        raise ValueError("边界必须使用整数圆环编号。")
    if boundaries[0] < 1 or boundaries[-1] >= RING_COUNT:
        raise ValueError("边界必须位于第 1 环至第 27 环。")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("五条内部边界必须严格递增。")


def boundary_perturbations() -> tuple[BoundaryPerturbation, ...]:
    """生成全部合法的单边界正负 1--2 环扰动。"""

    candidates: list[BoundaryPerturbation] = []
    for boundary_id, original in enumerate(BASE_BOUNDARIES, start=1):
        for shift in BOUNDARY_SHIFTS:
            values = list(BASE_BOUNDARIES)
            values[boundary_id - 1] = original + shift
            boundaries = tuple(values)
            try:
                validate_boundaries(boundaries)
            except ValueError:
                continue
            candidates.append(
                BoundaryPerturbation(
                    boundary_id=boundary_id,
                    shift_rings=shift,
                    boundaries=boundaries,
                )
            )
    return tuple(candidates)


def group_indices_for_boundaries(
    ring_indices: np.ndarray,
    boundaries: tuple[int, ...],
) -> np.ndarray:
    """按每组末环编号把逐镜圆环映射为从 0 开始的分区编号。"""

    validate_boundaries(boundaries)
    rings = np.asarray(ring_indices, dtype=np.int64)
    if rings.ndim != 1 or rings.size == 0:
        raise ValueError("ring_indices 必须是一维非空数组。")
    if int(np.min(rings)) < 1 or int(np.max(rings)) > RING_COUNT:
        raise ValueError("ring_indices 必须位于第 1 环至第 28 环。")
    return np.searchsorted(
        np.asarray(boundaries, dtype=np.int64),
        rings,
        side="left",
    ).astype(np.int64, copy=False)


def reassign_boundary_groups(
    field: RefineField,
    boundaries: tuple[int, ...],
) -> RefineField:
    """固定塔位与镜位，只更新逐镜所属分区。"""

    return replace(
        field,
        group_indices=group_indices_for_boundaries(
            field.ring_indices,
            boundaries,
        ),
    )


def moved_mirror_count(
    field: RefineField,
    boundaries: tuple[int, ...],
) -> int:
    assigned = group_indices_for_boundaries(field.ring_indices, boundaries)
    return int(np.count_nonzero(assigned != field.group_indices))


def specification_perturbations(
    design: RefineDesign,
    *,
    step_m: float = 0.10,
) -> tuple[Perturbation, ...]:
    if step_m <= 0.0:
        raise ValueError("敏感性扰动步长必须大于 0。")
    candidates: list[Perturbation] = []
    for parameter in SPECIFICATION_VARIABLES:
        old = design.parameter(parameter)
        for direction, delta in (("-", -step_m), ("+", step_m)):
            new = old + delta
            candidates.append(
                Perturbation(
                    parameter=parameter,
                    group_id=int(parameter[1:]),
                    direction=direction,
                    old_value=old,
                    new_value=new,
                    design=design.with_parameter(parameter, new),
                )
            )
    return tuple(candidates)


def select_formal_directions(
    rows: list[dict[str, object]],
    *,
    limit: int = 6,
) -> list[dict[str, object]]:
    """按中精度提升选出至多六个不同变量的最佳方向。"""

    eligible = [
        row
        for row in rows
        if bool(row.get("legal"))
        and row.get("medium_q") is not None
        and float(row["medium_power"]) >= 42.0
    ]
    eligible.sort(key=lambda row: float(row["delta_q"]), reverse=True)
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in eligible:
        parameter = str(row["parameter"])
        if parameter in seen:
            continue
        selected.append(row)
        seen.add(parameter)
        if len(selected) >= limit:
            break
    return selected


def active_from_formal(
    rows: list[dict[str, object]],
    *,
    reference_q: float,
    target_power_mw: float = 42.0,
    threshold: float = 1e-8,
) -> tuple[str, ...]:
    active = {
        str(row["parameter"])
        for row in rows
        if row.get("formal_q") is not None
        and float(row["formal_power"]) >= target_power_mw
        and float(row["formal_q"]) > reference_q + threshold
    }
    return tuple(
        parameter for parameter in SPECIFICATION_VARIABLES if parameter in active
    )
