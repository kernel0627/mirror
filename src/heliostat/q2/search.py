"""第二问两种布局的分散初值与循环变步长搜索。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Generic, Iterable, TypeVar

from scipy.stats import qmc

from .evaluate import (
    EvaluationCache,
    EvaluationProfile,
    ExtentScanResult,
    better_evaluation,
    scan_layout_extents,
)
from .layout import (
    CampoParameters,
    LayoutError,
    PartitionedRingParameters,
    generate_campo_layout,
    generate_partitioned_layout,
)


ParametersT = TypeVar(
    "ParametersT",
    PartitionedRingParameters,
    CampoParameters,
)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("参数边界必须为有限数。")
        if self.lower > self.upper:
            raise ValueError("参数下界不能大于上界。")

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def map_unit(self, value: float) -> float:
        return self.lower + value * (self.upper - self.lower)


@dataclass(frozen=True)
class IntegerInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("整数参数下界不能大于上界。")

    def contains(self, value: int) -> bool:
        return self.lower <= value <= self.upper

    def map_unit(self, value: float) -> int:
        mapped = self.lower + value * (self.upper - self.lower)
        return min(self.upper, max(self.lower, int(round(mapped))))


@dataclass(frozen=True)
class PartitionedBounds:
    tower_y: Interval = Interval(-220.0, 0.0)
    mirror_width: Interval = Interval(4.5, 7.5)
    mirror_height: Interval = Interval(4.0, 7.5)
    installation_height: Interval = Interval(2.0, 6.0)
    split_radius: Interval = Interval(150.0, 300.0)
    near_spacing: Interval = Interval(8.0, 22.0)
    far_spacing: Interval = Interval(10.0, 32.0)


@dataclass(frozen=True)
class CampoBounds:
    tower_y: Interval = Interval(-220.0, 0.0)
    mirror_width: Interval = Interval(4.5, 7.5)
    mirror_height: Interval = Interval(4.0, 7.5)
    installation_height: Interval = Interval(2.0, 6.0)
    first_ring_count: IntegerInterval = IntegerInterval(44, 76)
    initial_spacing: Interval = Interval(8.0, 22.0)
    spacing_growth: Interval = Interval(0.0, 0.6)


@dataclass(frozen=True)
class StepLevel:
    steps: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, float]:
        return dict(self.steps)


PARTITIONED_STEP_LEVELS = (
    StepLevel(
        (
            ("tower_y", 20.0),
            ("mirror_width", 0.5),
            ("mirror_height", 0.5),
            ("installation_height", 0.5),
            ("split_radius", 20.0),
            ("near_spacing", 2.0),
            ("far_spacing", 2.0),
        )
    ),
    StepLevel(
        (
            ("tower_y", 10.0),
            ("mirror_width", 0.2),
            ("mirror_height", 0.2),
            ("installation_height", 0.2),
            ("split_radius", 10.0),
            ("near_spacing", 1.0),
            ("far_spacing", 1.0),
        )
    ),
    StepLevel(
        (
            ("tower_y", 5.0),
            ("mirror_width", 0.1),
            ("mirror_height", 0.1),
            ("installation_height", 0.1),
            ("split_radius", 5.0),
            ("near_spacing", 0.5),
            ("far_spacing", 0.5),
        )
    ),
)


CAMPO_STEP_LEVELS = (
    StepLevel(
        (
            ("tower_y", 20.0),
            ("mirror_width", 0.5),
            ("mirror_height", 0.5),
            ("installation_height", 0.5),
            ("first_ring_count", 4.0),
            ("initial_spacing", 2.0),
            ("spacing_growth", 0.1),
        )
    ),
    StepLevel(
        (
            ("tower_y", 10.0),
            ("mirror_width", 0.2),
            ("mirror_height", 0.2),
            ("installation_height", 0.2),
            ("first_ring_count", 2.0),
            ("initial_spacing", 1.0),
            ("spacing_growth", 0.05),
        )
    ),
    StepLevel(
        (
            ("tower_y", 5.0),
            ("mirror_width", 0.1),
            ("mirror_height", 0.1),
            ("installation_height", 0.1),
            ("first_ring_count", 1.0),
            ("initial_spacing", 0.5),
            ("spacing_growth", 0.02),
        )
    ),
)


@dataclass(frozen=True)
class SearchOutcome(Generic[ParametersT]):
    parameters: ParametersT
    extent_scan: ExtentScanResult

    @property
    def feasible(self) -> bool:
        return self.extent_scan.best.is_feasible()

    @property
    def annual_power_mw(self) -> float:
        return self.extent_scan.best.annual_power_mw

    @property
    def unit_area_power_kw_m2(self) -> float:
        return self.extent_scan.best.unit_area_power_kw_m2


@dataclass(frozen=True)
class SearchTrace(Generic[ParametersT]):
    step_level: int
    cycle: int
    block: str
    outcome: SearchOutcome[ParametersT]


@dataclass(frozen=True)
class OptimizationResult(Generic[ParametersT]):
    layout_kind: str
    best: SearchOutcome[ParametersT]
    starts: tuple[SearchOutcome[ParametersT], ...]
    local_results: tuple[SearchOutcome[ParametersT], ...]
    trace: tuple[SearchTrace[ParametersT], ...]


def _sobol_points(dimension: int, count: int, seed: int) -> list[list[float]]:
    if count < 1:
        raise ValueError("Sobol 样本数必须大于等于 1。")
    exponent = int(math.ceil(math.log2(count)))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    return sampler.random_base2(exponent)[:count].tolist()


def sample_partitioned_parameters(
    count: int,
    *,
    seed: int = 2023,
    bounds: PartitionedBounds = PartitionedBounds(),
) -> tuple[PartitionedRingParameters, ...]:
    samples: list[PartitionedRingParameters] = []
    for point in _sobol_points(7, count, seed):
        width = bounds.mirror_width.map_unit(point[1])
        height_upper = min(bounds.mirror_height.upper, width)
        height_lower = min(bounds.mirror_height.lower, height_upper)
        height = height_lower + point[2] * (height_upper - height_lower)
        installation_lower = max(
            bounds.installation_height.lower,
            height / 2.0,
        )
        installation = installation_lower + point[3] * (
            bounds.installation_height.upper - installation_lower
        )
        near = bounds.near_spacing.map_unit(point[5])
        far_lower = max(bounds.far_spacing.lower, near)
        far = far_lower + point[6] * (bounds.far_spacing.upper - far_lower)
        samples.append(
            PartitionedRingParameters(
                tower_y=bounds.tower_y.map_unit(point[0]),
                mirror_width=width,
                mirror_height=height,
                installation_height=installation,
                split_radius=bounds.split_radius.map_unit(point[4]),
                near_spacing=near,
                far_spacing=far,
            )
        )
    return tuple(samples)


def sample_campo_parameters(
    count: int,
    *,
    seed: int = 2023,
    bounds: CampoBounds = CampoBounds(),
) -> tuple[CampoParameters, ...]:
    samples: list[CampoParameters] = []
    for point in _sobol_points(7, count, seed):
        width = bounds.mirror_width.map_unit(point[1])
        height_upper = min(bounds.mirror_height.upper, width)
        height_lower = min(bounds.mirror_height.lower, height_upper)
        height = height_lower + point[2] * (height_upper - height_lower)
        installation_lower = max(
            bounds.installation_height.lower,
            height / 2.0,
        )
        installation = installation_lower + point[3] * (
            bounds.installation_height.upper - installation_lower
        )
        samples.append(
            CampoParameters(
                tower_y=bounds.tower_y.map_unit(point[0]),
                mirror_width=width,
                mirror_height=height,
                installation_height=installation,
                first_ring_count=bounds.first_ring_count.map_unit(point[4]),
                initial_spacing=bounds.initial_spacing.map_unit(point[5]),
                spacing_growth=bounds.spacing_growth.map_unit(point[6]),
            )
        )
    return tuple(samples)


def _partitioned_within_bounds(
    parameters: PartitionedRingParameters,
    bounds: PartitionedBounds,
) -> bool:
    return (
        bounds.tower_y.contains(parameters.tower_y)
        and bounds.mirror_width.contains(parameters.mirror_width)
        and bounds.mirror_height.contains(parameters.mirror_height)
        and parameters.mirror_height <= parameters.mirror_width
        and bounds.installation_height.contains(parameters.installation_height)
        and parameters.installation_height >= parameters.mirror_height / 2.0
        and bounds.split_radius.contains(parameters.split_radius)
        and bounds.near_spacing.contains(parameters.near_spacing)
        and bounds.far_spacing.contains(parameters.far_spacing)
        and parameters.far_spacing >= parameters.near_spacing
    )


def _campo_within_bounds(
    parameters: CampoParameters,
    bounds: CampoBounds,
) -> bool:
    return (
        bounds.tower_y.contains(parameters.tower_y)
        and bounds.mirror_width.contains(parameters.mirror_width)
        and bounds.mirror_height.contains(parameters.mirror_height)
        and parameters.mirror_height <= parameters.mirror_width
        and bounds.installation_height.contains(parameters.installation_height)
        and parameters.installation_height >= parameters.mirror_height / 2.0
        and bounds.first_ring_count.contains(parameters.first_ring_count)
        and bounds.initial_spacing.contains(parameters.initial_spacing)
        and bounds.spacing_growth.contains(parameters.spacing_growth)
    )


def _better_outcome(
    left: SearchOutcome[ParametersT],
    right: SearchOutcome[ParametersT],
) -> SearchOutcome[ParametersT]:
    best_evaluation = better_evaluation(
        left.extent_scan.best,
        right.extent_scan.best,
    )
    return left if best_evaluation is left.extent_scan.best else right


def _rank_outcomes(
    outcomes: Iterable[SearchOutcome[ParametersT]],
) -> list[SearchOutcome[ParametersT]]:
    ranked: list[SearchOutcome[ParametersT]] = []
    for outcome in outcomes:
        inserted = False
        for index, current in enumerate(ranked):
            if _better_outcome(outcome, current) is outcome:
                ranked.insert(index, outcome)
                inserted = True
                break
        if not inserted:
            ranked.append(outcome)
    return ranked


def evaluate_partitioned_parameters(
    parameters: PartitionedRingParameters,
    profile: EvaluationProfile,
    *,
    cache: EvaluationCache | None = None,
    coarse_stride: int = 4,
    window: int = 2,
) -> SearchOutcome[PartitionedRingParameters]:
    layout = generate_partitioned_layout(parameters)
    scan = scan_layout_extents(
        layout,
        parameters,
        profile,
        coarse_stride=coarse_stride,
        window=window,
        cache=cache,
    )
    return SearchOutcome(parameters, scan)


def evaluate_campo_parameters(
    parameters: CampoParameters,
    profile: EvaluationProfile,
    *,
    cache: EvaluationCache | None = None,
    coarse_stride: int = 4,
    window: int = 2,
) -> SearchOutcome[CampoParameters]:
    layout = generate_campo_layout(parameters)
    scan = scan_layout_extents(
        layout,
        parameters,
        profile,
        coarse_stride=coarse_stride,
        window=window,
        cache=cache,
    )
    return SearchOutcome(parameters, scan)


def _coordinate_candidates(
    parameters: ParametersT,
    fields: tuple[str, ...],
    steps: dict[str, float],
) -> tuple[ParametersT, ...]:
    candidates: list[ParametersT] = []
    for field in fields:
        step = steps[field]
        value = getattr(parameters, field)
        for direction in (-1.0, 1.0):
            new_value: float | int = value + direction * step
            if isinstance(value, int):
                new_value = int(round(new_value))
            candidates.append(replace(parameters, **{field: new_value}))
    return tuple(candidates)


def _cyclic_search(
    initial: SearchOutcome[ParametersT],
    *,
    evaluator: Callable[[ParametersT], SearchOutcome[ParametersT]],
    within_bounds: Callable[[ParametersT], bool],
    blocks: tuple[tuple[str, tuple[str, ...]], ...],
    step_levels: tuple[StepLevel, ...],
    maximum_cycles_per_level: int,
) -> tuple[SearchOutcome[ParametersT], tuple[SearchTrace[ParametersT], ...]]:
    current = initial
    trace: list[SearchTrace[ParametersT]] = []

    for level_index, level in enumerate(step_levels):
        steps = level.as_dict()
        for cycle in range(maximum_cycles_per_level):
            cycle_improved = False
            for block_name, fields in blocks:
                block_best = current
                for candidate in _coordinate_candidates(
                    current.parameters,
                    fields,
                    steps,
                ):
                    if not within_bounds(candidate):
                        continue
                    try:
                        outcome = evaluator(candidate)
                    except LayoutError:
                        continue
                    block_best = _better_outcome(block_best, outcome)
                if block_best is not current:
                    current = block_best
                    cycle_improved = True
                    trace.append(
                        SearchTrace(
                            step_level=level_index,
                            cycle=cycle,
                            block=block_name,
                            outcome=current,
                        )
                    )
            if not cycle_improved:
                break
    return current, tuple(trace)


def optimize_partitioned(
    *,
    profile: EvaluationProfile,
    initial_sample_count: int = 16,
    retained_starts: int = 3,
    seed: int = 2023,
    bounds: PartitionedBounds = PartitionedBounds(),
    step_levels: tuple[StepLevel, ...] = PARTITIONED_STEP_LEVELS,
    maximum_cycles_per_level: int = 4,
    coarse_stride: int = 4,
    window: int = 2,
    cache: EvaluationCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult[PartitionedRingParameters]:
    cache = cache or EvaluationCache()
    outcomes: list[SearchOutcome[PartitionedRingParameters]] = []
    samples = sample_partitioned_parameters(
        initial_sample_count,
        seed=seed,
        bounds=bounds,
    )
    for index, parameters in enumerate(samples, start=1):
        try:
            outcome = evaluate_partitioned_parameters(
                parameters,
                profile,
                cache=cache,
                coarse_stride=coarse_stride,
                window=window,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(
                    f"方案 A 初值 {index}/{len(samples)}："
                    f"P={outcome.annual_power_mw:.4f} MW，"
                    f"q={outcome.unit_area_power_kw_m2:.6f}"
                )
        except LayoutError as exc:
            if progress is not None:
                progress(f"方案 A 初值 {index}/{len(samples)}：几何不可行（{exc}）")
            continue
    if not outcomes:
        raise RuntimeError("分区圆环的分散初值中没有几何合法方案。")

    starts = tuple(_rank_outcomes(outcomes)[:retained_starts])
    local_results: list[SearchOutcome[PartitionedRingParameters]] = []
    trace: list[SearchTrace[PartitionedRingParameters]] = []

    def evaluator(
        parameters: PartitionedRingParameters,
    ) -> SearchOutcome[PartitionedRingParameters]:
        return evaluate_partitioned_parameters(
            parameters,
            profile,
            cache=cache,
            coarse_stride=coarse_stride,
            window=window,
        )

    blocks = (
        ("tower", ("tower_y",)),
        (
            "mirror",
            ("mirror_width", "mirror_height", "installation_height"),
        ),
        ("layout", ("split_radius", "near_spacing", "far_spacing")),
    )
    for index, start in enumerate(starts, start=1):
        if progress is not None:
            progress(f"方案 A 开始局部搜索 {index}/{len(starts)}")
        local, local_trace = _cyclic_search(
            start,
            evaluator=evaluator,
            within_bounds=lambda value: _partitioned_within_bounds(
                value,
                bounds,
            ),
            blocks=blocks,
            step_levels=step_levels,
            maximum_cycles_per_level=maximum_cycles_per_level,
        )
        local_results.append(local)
        trace.extend(local_trace)
    best = _rank_outcomes(local_results)[0]
    return OptimizationResult(
        "partitioned",
        best,
        starts,
        tuple(local_results),
        tuple(trace),
    )


def refine_partitioned(
    initial_parameters: PartitionedRingParameters,
    *,
    profile: EvaluationProfile,
    bounds: PartitionedBounds = PartitionedBounds(),
    step_levels: tuple[StepLevel, ...] = PARTITIONED_STEP_LEVELS,
    maximum_cycles_per_level: int = 4,
    coarse_stride: int = 4,
    window: int = 2,
    cache: EvaluationCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult[PartitionedRingParameters]:
    """从已落盘的方案 A 参数继续循环变步长搜索。"""

    if not _partitioned_within_bounds(initial_parameters, bounds):
        raise ValueError("方案 A 的恢复参数超出当前搜索边界。")
    cache = cache or EvaluationCache()

    def evaluator(
        parameters: PartitionedRingParameters,
    ) -> SearchOutcome[PartitionedRingParameters]:
        return evaluate_partitioned_parameters(
            parameters,
            profile,
            cache=cache,
            coarse_stride=coarse_stride,
            window=window,
        )

    initial = evaluator(initial_parameters)
    if progress is not None:
        progress(
            "方案 A 恢复起点："
            f"P={initial.annual_power_mw:.4f} MW，"
            f"q={initial.unit_area_power_kw_m2:.6f}"
        )
    blocks = (
        ("tower", ("tower_y",)),
        (
            "mirror",
            ("mirror_width", "mirror_height", "installation_height"),
        ),
        ("layout", ("split_radius", "near_spacing", "far_spacing")),
    )
    best, trace = _cyclic_search(
        initial,
        evaluator=evaluator,
        within_bounds=lambda value: _partitioned_within_bounds(value, bounds),
        blocks=blocks,
        step_levels=step_levels,
        maximum_cycles_per_level=maximum_cycles_per_level,
    )
    return OptimizationResult(
        "partitioned",
        best,
        (initial,),
        (best,),
        trace,
    )


def optimize_campo(
    *,
    profile: EvaluationProfile,
    initial_sample_count: int = 16,
    retained_starts: int = 3,
    seed: int = 2023,
    bounds: CampoBounds = CampoBounds(),
    step_levels: tuple[StepLevel, ...] = CAMPO_STEP_LEVELS,
    maximum_cycles_per_level: int = 4,
    coarse_stride: int = 4,
    window: int = 2,
    cache: EvaluationCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult[CampoParameters]:
    cache = cache or EvaluationCache()
    outcomes: list[SearchOutcome[CampoParameters]] = []
    samples = sample_campo_parameters(
        initial_sample_count,
        seed=seed,
        bounds=bounds,
    )
    for index, parameters in enumerate(samples, start=1):
        try:
            outcome = evaluate_campo_parameters(
                parameters,
                profile,
                cache=cache,
                coarse_stride=coarse_stride,
                window=window,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(
                    f"方案 B 初值 {index}/{len(samples)}："
                    f"P={outcome.annual_power_mw:.4f} MW，"
                    f"q={outcome.unit_area_power_kw_m2:.6f}"
                )
        except LayoutError as exc:
            if progress is not None:
                progress(f"方案 B 初值 {index}/{len(samples)}：几何不可行（{exc}）")
            continue
    if not outcomes:
        raise RuntimeError("Campo 的分散初值中没有几何合法方案。")

    starts = tuple(_rank_outcomes(outcomes)[:retained_starts])
    local_results: list[SearchOutcome[CampoParameters]] = []
    trace: list[SearchTrace[CampoParameters]] = []

    def evaluator(
        parameters: CampoParameters,
    ) -> SearchOutcome[CampoParameters]:
        return evaluate_campo_parameters(
            parameters,
            profile,
            cache=cache,
            coarse_stride=coarse_stride,
            window=window,
        )

    blocks = (
        ("tower", ("tower_y",)),
        (
            "mirror",
            ("mirror_width", "mirror_height", "installation_height"),
        ),
        ("layout", ("first_ring_count", "initial_spacing", "spacing_growth")),
    )
    for start in starts:
        local, local_trace = _cyclic_search(
            start,
            evaluator=evaluator,
            within_bounds=lambda value: _campo_within_bounds(value, bounds),
            blocks=blocks,
            step_levels=step_levels,
            maximum_cycles_per_level=maximum_cycles_per_level,
        )
        local_results.append(local)
        trace.extend(local_trace)
    best = _rank_outcomes(local_results)[0]
    return OptimizationResult(
        "campo",
        best,
        starts,
        tuple(local_results),
        tuple(trace),
    )


def refine_campo(
    initial_parameters: CampoParameters,
    *,
    profile: EvaluationProfile,
    bounds: CampoBounds = CampoBounds(),
    step_levels: tuple[StepLevel, ...] = CAMPO_STEP_LEVELS,
    maximum_cycles_per_level: int = 4,
    coarse_stride: int = 4,
    window: int = 2,
    cache: EvaluationCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult[CampoParameters]:
    """从已落盘的方案 B 参数继续循环变步长搜索。"""

    if not _campo_within_bounds(initial_parameters, bounds):
        raise ValueError("方案 B 的恢复参数超出当前搜索边界。")
    cache = cache or EvaluationCache()

    def evaluator(
        parameters: CampoParameters,
    ) -> SearchOutcome[CampoParameters]:
        return evaluate_campo_parameters(
            parameters,
            profile,
            cache=cache,
            coarse_stride=coarse_stride,
            window=window,
        )

    initial = evaluator(initial_parameters)
    if progress is not None:
        progress(
            "方案 B 恢复起点："
            f"P={initial.annual_power_mw:.4f} MW，"
            f"q={initial.unit_area_power_kw_m2:.6f}"
        )
    blocks = (
        ("tower", ("tower_y",)),
        (
            "mirror",
            ("mirror_width", "mirror_height", "installation_height"),
        ),
        ("layout", ("first_ring_count", "initial_spacing", "spacing_growth")),
    )
    best, trace = _cyclic_search(
        initial,
        evaluator=evaluator,
        within_bounds=lambda value: _campo_within_bounds(value, bounds),
        blocks=blocks,
        step_levels=step_levels,
        maximum_cycles_per_level=maximum_cycles_per_level,
    )
    return OptimizationResult(
        "campo",
        best,
        (initial,),
        (best,),
        trace,
    )
