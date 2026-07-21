"""六区 18 个规格变量的正负扰动与活跃变量筛选。"""

from __future__ import annotations

from dataclasses import dataclass

from .model import RefineDesign


SPECIFICATION_VARIABLES = tuple(
    f"{prefix}{group}"
    for prefix in ("w", "h", "H")
    for group in range(1, 7)
)


@dataclass(frozen=True)
class Perturbation:
    parameter: str
    group_id: int
    direction: str
    old_value: float
    new_value: float
    design: RefineDesign


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
