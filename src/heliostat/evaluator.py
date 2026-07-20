"""第一问的逐时刻计算、汇总与正确性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import SolverConfig
from .geometry import (
    PreparedField,
    calculate_orientation,
    maximum_reflection_error,
)
from .shadow import calculate_shadow_blocking_efficiency
from .solar import calculate_solar_state
from .truncation import calculate_truncation_efficiency


SOLAR_TIMES = (9.0, 10.5, 12.0, 13.5, 15.0)


@dataclass(frozen=True)
class TimeResult:
    month: int
    solar_time: float
    dni_kw_m2: float
    average_optical_efficiency: float
    average_cosine_efficiency: float
    average_shadow_blocking_efficiency: float
    average_atmospheric_efficiency: float
    average_truncation_efficiency: float
    field_output_mw: float
    unit_area_output_kw_m2: float
    maximum_reflection_error: float


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
class Question1Solution:
    time_results: tuple[TimeResult, ...]
    monthly_results: tuple[MonthlyResult, ...]
    annual_result: AnnualResult


ProgressCallback = Callable[[int, int, TimeResult], None]


def _check_efficiency(name: str, values: np.ndarray) -> None:
    tolerance = 1e-12
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        raise RuntimeError(
            f"{name} 超出 [0, 1]：min={minimum:.6g}, max={maximum:.6g}"
        )


def evaluate_time(
    prepared: PreparedField,
    month: int,
    solar_time: float,
    solver: SolverConfig,
) -> TimeResult:
    solar = calculate_solar_state(
        month=month,
        solar_time=solar_time,
        latitude_deg=prepared.config.latitude_deg,
        altitude_km=prepared.config.altitude_km,
    )
    orientation = calculate_orientation(prepared, solar.direction)
    reflection_error = maximum_reflection_error(
        prepared,
        orientation,
        solar.direction,
    )
    if reflection_error >= 1e-8:
        raise RuntimeError(
            f"中心光线反射误差过大：{reflection_error:.3e}"
        )

    if solver.calculate_shadow:
        shadow = calculate_shadow_blocking_efficiency(
            prepared,
            orientation,
            solar.direction,
            solver,
        )
    else:
        shadow = np.ones(prepared.mirror_count, dtype=float)

    if solver.calculate_truncation:
        truncation = calculate_truncation_efficiency(
            prepared,
            orientation,
            solar.direction,
            solver,
        )
    else:
        truncation = np.ones(prepared.mirror_count, dtype=float)

    cosine = orientation.cosine_efficiency
    atmospheric = prepared.atmospheric_efficiency
    optical = (
        cosine
        * shadow
        * atmospheric
        * truncation
        * prepared.config.reflectivity
    )
    for name, values in (
        ("余弦效率", cosine),
        ("阴影遮挡效率", shadow),
        ("大气透射率", atmospheric),
        ("截断效率", truncation),
        ("光学效率", optical),
    ):
        _check_efficiency(name, values)

    mirror_power_kw = solar.dni_kw_m2 * prepared.config.mirror_area * optical
    field_power_kw = float(np.sum(mirror_power_kw))
    total_area = prepared.total_mirror_area

    return TimeResult(
        month=month,
        solar_time=solar_time,
        dni_kw_m2=solar.dni_kw_m2,
        average_optical_efficiency=float(np.mean(optical)),
        average_cosine_efficiency=float(np.mean(cosine)),
        average_shadow_blocking_efficiency=float(np.mean(shadow)),
        average_atmospheric_efficiency=float(np.mean(atmospheric)),
        average_truncation_efficiency=float(np.mean(truncation)),
        field_output_mw=field_power_kw / 1000.0,
        unit_area_output_kw_m2=field_power_kw / total_area,
        maximum_reflection_error=reflection_error,
    )


def _means(
    records: Sequence[TimeResult],
) -> tuple[float, float, float, float, float, float, float]:
    if not records:
        raise ValueError("汇总记录不能为空。")
    return tuple(
        float(np.mean([getattr(record, field) for record in records]))
        for field in (
            "average_optical_efficiency",
            "average_cosine_efficiency",
            "average_shadow_blocking_efficiency",
            "average_atmospheric_efficiency",
            "average_truncation_efficiency",
            "field_output_mw",
            "unit_area_output_kw_m2",
        )
    )


def summarize_monthly(records: Sequence[TimeResult]) -> tuple[MonthlyResult, ...]:
    results: list[MonthlyResult] = []
    for month in sorted({record.month for record in records}):
        monthly = [record for record in records if record.month == month]
        values = _means(monthly)
        results.append(MonthlyResult(month, *values))
    return tuple(results)


def summarize_annual(records: Sequence[TimeResult]) -> AnnualResult:
    return AnnualResult(*_means(records))


def solve_question1(
    prepared: PreparedField,
    solver: SolverConfig,
    months: Sequence[int] = tuple(range(1, 13)),
    solar_times: Sequence[float] = SOLAR_TIMES,
    progress: ProgressCallback | None = None,
) -> Question1Solution:
    """执行所选月份和时刻；默认即题目规定的 60 个状态。"""

    if not months or not solar_times:
        raise ValueError("months 和 solar_times 不能为空。")
    if any(month < 1 or month > 12 for month in months):
        raise ValueError("months 必须位于 1 到 12。")

    records: list[TimeResult] = []
    total = len(months) * len(solar_times)
    for month in months:
        for solar_time in solar_times:
            record = evaluate_time(prepared, month, solar_time, solver)
            records.append(record)
            if progress is not None:
                progress(len(records), total, record)

    time_results = tuple(records)
    return Question1Solution(
        time_results=time_results,
        monthly_results=summarize_monthly(time_results),
        annual_result=summarize_annual(time_results),
    )
