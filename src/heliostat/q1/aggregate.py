"""第一问的月平均和年平均汇总。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MonthlyResult:
    month: int
    average_optical_efficiency: float
    average_cosine_efficiency: float
    average_shadow_blocking_efficiency: float
    average_atmospheric_efficiency: float
    average_truncation_efficiency: float
    field_output_mw: float
    unit_area_output_kw_m2: float


@dataclass(frozen=True)
class AnnualResult:
    average_optical_efficiency: float
    average_cosine_efficiency: float
    average_shadow_blocking_efficiency: float
    average_atmospheric_efficiency: float
    average_truncation_efficiency: float
    field_output_mw: float
    unit_area_output_kw_m2: float


@dataclass(frozen=True)
class MirrorAnnualResult:
    mirror_id: int
    x_m: float
    y_m: float
    radius_to_tower_m: float
    average_optical_efficiency: float
    average_cosine_efficiency: float
    average_shadow_blocking_efficiency: float
    average_atmospheric_efficiency: float
    average_truncation_efficiency: float
    average_output_power_kw: float


@dataclass(frozen=True)
class Question1Solution:
    time_results: tuple[Any, ...]
    monthly_results: tuple[MonthlyResult, ...]
    annual_result: AnnualResult
    mirror_annual_results: tuple[MirrorAnnualResult, ...]


_MEAN_FIELDS = (
    "average_optical_efficiency",
    "average_cosine_efficiency",
    "average_shadow_blocking_efficiency",
    "average_atmospheric_efficiency",
    "average_truncation_efficiency",
    "field_output_mw",
    "unit_area_output_kw_m2",
)


def _means(records: Sequence[Any]) -> tuple[float, ...]:
    if not records:
        raise ValueError("汇总记录不能为空。")
    return tuple(
        float(np.mean([getattr(record, field) for record in records]))
        for field in _MEAN_FIELDS
    )


def summarize_monthly(records: Sequence[Any]) -> tuple[MonthlyResult, ...]:
    """对每月规定的五个时刻等权平均。"""

    results: list[MonthlyResult] = []
    for month in sorted({record.month for record in records}):
        monthly = [record for record in records if record.month == month]
        results.append(MonthlyResult(month, *_means(monthly)))
    return tuple(results)


def summarize_annual(records: Sequence[Any]) -> AnnualResult:
    """对题目规定的全部时刻等权平均。"""

    return AnnualResult(*_means(records))


def summarize_mirror_annual(
    mirror_xy: np.ndarray,
    tower_x: float,
    tower_y: float,
    state_count: int,
    optical_efficiency_sum: np.ndarray,
    cosine_efficiency_sum: np.ndarray,
    shadow_blocking_efficiency_sum: np.ndarray,
    atmospheric_efficiency_sum: np.ndarray,
    truncation_efficiency_sum: np.ndarray,
    output_power_kw_sum: np.ndarray,
) -> tuple[MirrorAnnualResult, ...]:
    """由逐时刻运行和生成单镜年平均结果，不保留单镜逐时刻明细。"""

    if state_count < 1:
        raise ValueError("state_count 必须大于等于 1。")
    radius = np.hypot(mirror_xy[:, 0] - tower_x, mirror_xy[:, 1] - tower_y)
    means = {
        "optical": optical_efficiency_sum / state_count,
        "cosine": cosine_efficiency_sum / state_count,
        "shadow": shadow_blocking_efficiency_sum / state_count,
        "atmospheric": atmospheric_efficiency_sum / state_count,
        "truncation": truncation_efficiency_sum / state_count,
        "power": output_power_kw_sum / state_count,
    }
    return tuple(
        MirrorAnnualResult(
            mirror_id=index + 1,
            x_m=float(mirror_xy[index, 0]),
            y_m=float(mirror_xy[index, 1]),
            radius_to_tower_m=float(radius[index]),
            average_optical_efficiency=float(means["optical"][index]),
            average_cosine_efficiency=float(means["cosine"][index]),
            average_shadow_blocking_efficiency=float(means["shadow"][index]),
            average_atmospheric_efficiency=float(
                means["atmospheric"][index]
            ),
            average_truncation_efficiency=float(means["truncation"][index]),
            average_output_power_kw=float(means["power"][index]),
        )
        for index in range(mirror_xy.shape[0])
    )
