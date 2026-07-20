"""第二问的两种参数化镜场布局与统一几何约束检查。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]


class LayoutError(ValueError):
    """布局参数或生成结果不可行。"""


class CommonParameters(Protocol):
    tower_x: float
    tower_y: float
    mirror_width: float
    mirror_height: float
    installation_height: float
    field_radius: float
    exclusion_radius: float
    safety_epsilon: float


@dataclass(frozen=True)
class PartitionedRingParameters:
    """分区交错同心圆布局参数。"""

    tower_y: float
    mirror_width: float
    mirror_height: float
    installation_height: float
    split_radius: float
    near_spacing: float
    far_spacing: float
    tower_x: float = 0.0
    field_radius: float = 350.0
    exclusion_radius: float = 100.0
    safety_epsilon: float = 0.01

    @property
    def safe_distance(self) -> float:
        return self.mirror_width + 5.0 + self.safety_epsilon


@dataclass(frozen=True)
class CampoParameters:
    """带渐增径向行距的 Campo 径向交错布局参数。"""

    tower_y: float
    mirror_width: float
    mirror_height: float
    installation_height: float
    first_ring_count: int
    initial_spacing: float
    spacing_growth: float
    tower_x: float = 0.0
    field_radius: float = 350.0
    exclusion_radius: float = 100.0
    safety_epsilon: float = 0.01

    @property
    def safe_distance(self) -> float:
        return self.mirror_width + 5.0 + self.safety_epsilon


@dataclass(frozen=True)
class LayoutRing:
    """一条生成并经过场地裁剪的圆环或圆弧。"""

    index: int
    radius: float
    zone: int
    nominal_count: int
    coordinates: FloatArray

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])


@dataclass(frozen=True)
class GeneratedLayout:
    """由若干按半径排序的圆环或圆弧组成的镜场。"""

    kind: str
    rings: tuple[LayoutRing, ...]

    @property
    def mirror_count(self) -> int:
        return sum(ring.mirror_count for ring in self.rings)

    @property
    def coordinates(self) -> FloatArray:
        if not self.rings:
            return np.empty((0, 2), dtype=float)
        return np.concatenate(
            [ring.coordinates for ring in self.rings],
            axis=0,
        )

    def prefix(self, ring_count: int) -> FloatArray:
        if ring_count < 1 or ring_count > len(self.rings):
            raise ValueError(
                f"ring_count 应位于 1 到 {len(self.rings)}，实际为 {ring_count}。"
            )
        return np.concatenate(
            [ring.coordinates for ring in self.rings[:ring_count]],
            axis=0,
        )

    def prefix_mirror_count(self, ring_count: int) -> int:
        if ring_count < 1 or ring_count > len(self.rings):
            raise ValueError(
                f"ring_count 应位于 1 到 {len(self.rings)}，实际为 {ring_count}。"
            )
        return sum(ring.mirror_count for ring in self.rings[:ring_count])


@dataclass(frozen=True)
class GeometryCheck:
    valid: bool
    reason: str | None
    mirror_count: int
    minimum_center_distance: float
    maximum_field_radius: float
    minimum_tower_distance: float


def _validate_common_parameters(parameters: CommonParameters) -> None:
    values = (
        parameters.tower_x,
        parameters.tower_y,
        parameters.mirror_width,
        parameters.mirror_height,
        parameters.installation_height,
        parameters.field_radius,
        parameters.exclusion_radius,
        parameters.safety_epsilon,
    )
    if not all(math.isfinite(value) for value in values):
        raise LayoutError("布局参数必须全部为有限数。")
    if not 2.0 <= parameters.mirror_height <= parameters.mirror_width <= 8.0:
        raise LayoutError("镜面尺寸必须满足 2 ≤ h ≤ w ≤ 8。")
    if not 2.0 <= parameters.installation_height <= 6.0:
        raise LayoutError("安装高度必须位于 2 m 到 6 m。")
    if parameters.installation_height < parameters.mirror_height / 2.0:
        raise LayoutError("安装高度不足，镜面转动时可能触地。")
    if parameters.field_radius <= 0.0:
        raise LayoutError("场地半径必须大于 0。")
    if parameters.exclusion_radius <= 0.0:
        raise LayoutError("塔周禁区半径必须大于 0。")
    if parameters.safety_epsilon <= 0.0:
        raise LayoutError("安全距离余量必须大于 0。")


def _maximum_tower_centered_radius(parameters: CommonParameters) -> float:
    return parameters.field_radius + math.hypot(
        parameters.tower_x,
        parameters.tower_y,
    )


def _ring_coordinates(
    parameters: CommonParameters,
    radius: float,
    mirror_count: int,
    phase: float,
) -> FloatArray:
    if mirror_count < 2:
        raise LayoutError("单圈镜子数必须大于等于 2。")
    angles = 2.0 * math.pi * np.arange(mirror_count, dtype=float) / mirror_count + phase
    coordinates = np.column_stack(
        (
            parameters.tower_x + radius * np.sin(angles),
            parameters.tower_y + radius * np.cos(angles),
        )
    )
    field_radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    keep = field_radius <= parameters.field_radius + 1e-9
    clipped = np.asarray(coordinates[keep], dtype=float)
    clipped.setflags(write=False)
    return clipped


def _within_ring_count(radius: float, safe_distance: float) -> int:
    ratio = safe_distance / (2.0 * radius)
    if ratio >= 1.0:
        raise LayoutError("圆环半径过小，无法放置满足安全距离的镜子。")
    return int(math.floor(math.pi / math.asin(ratio)))


def generate_partitioned_layout(
    parameters: PartitionedRingParameters,
    *,
    maximum_rings: int = 256,
) -> GeneratedLayout:
    """生成分区交错同心圆，并拒绝跨环距离冲突。"""

    _validate_common_parameters(parameters)
    if not math.isfinite(parameters.split_radius):
        raise LayoutError("分区半径必须为有限数。")
    if parameters.split_radius <= parameters.exclusion_radius:
        raise LayoutError("分区半径必须位于塔周禁区之外。")
    if parameters.near_spacing <= 0.0:
        raise LayoutError("近区行距必须大于 0。")
    if parameters.far_spacing < parameters.near_spacing:
        raise LayoutError("远区行距必须大于等于近区行距。")

    rings: list[LayoutRing] = []
    radius = parameters.exclusion_radius
    maximum_radius = _maximum_tower_centered_radius(parameters)
    for ring_index in range(maximum_rings):
        if radius > maximum_radius + 1e-9:
            break
        count = _within_ring_count(radius, parameters.safe_distance)
        phase = 0.0 if ring_index % 2 == 0 else math.pi / count
        coordinates = _ring_coordinates(
            parameters,
            radius,
            count,
            phase,
        )
        if coordinates.size:
            rings.append(
                LayoutRing(
                    index=ring_index,
                    radius=radius,
                    zone=1 if radius < parameters.split_radius else 2,
                    nominal_count=count,
                    coordinates=coordinates,
                )
            )
        spacing = (
            parameters.near_spacing
            if radius < parameters.split_radius
            else parameters.far_spacing
        )
        radius += spacing
    else:
        raise LayoutError("达到 maximum_rings，圆环生成未正常终止。")

    layout = GeneratedLayout("partitioned", tuple(rings))
    check = validate_layout(layout.coordinates, parameters)
    if not check.valid:
        raise LayoutError(check.reason or "分区圆环布局不满足几何约束。")
    return layout


def _campo_zone(radius: float, first_radius: float) -> tuple[int, int]:
    if radius < 2.0 * first_radius:
        return 1, 1
    if radius < 4.0 * first_radius:
        return 2, 2
    return 3, 4


def generate_campo_layout(
    parameters: CampoParameters,
    *,
    maximum_rings: int = 256,
) -> GeneratedLayout:
    """生成三分区、渐增径向行距的 Campo 径向交错镜场。"""

    _validate_common_parameters(parameters)
    if parameters.first_ring_count < 2:
        raise LayoutError("Campo 首环镜子数必须大于等于 2。")
    if parameters.initial_spacing <= 0.0:
        raise LayoutError("Campo 初始行距必须大于 0。")
    if parameters.spacing_growth < 0.0:
        raise LayoutError("Campo 行距增长量不能小于 0。")

    first_radius = max(
        parameters.exclusion_radius,
        parameters.safe_distance
        / (2.0 * math.sin(math.pi / parameters.first_ring_count)),
    )
    maximum_radius = _maximum_tower_centered_radius(parameters)
    rings: list[LayoutRing] = []
    zone_rows = {1: 0, 2: 0, 3: 0}
    radius = first_radius

    for ring_index in range(maximum_rings):
        if radius > maximum_radius + 1e-9:
            break
        zone, multiplier = _campo_zone(radius, first_radius)
        count = parameters.first_ring_count * multiplier
        zone_index = zone_rows[zone]
        phase = 0.0 if zone_index % 2 == 0 else math.pi / count
        coordinates = _ring_coordinates(
            parameters,
            radius,
            count,
            phase,
        )
        if coordinates.size:
            rings.append(
                LayoutRing(
                    index=ring_index,
                    radius=radius,
                    zone=zone,
                    nominal_count=count,
                    coordinates=coordinates,
                )
            )
        zone_rows[zone] += 1
        radius += parameters.initial_spacing + parameters.spacing_growth * ring_index
    else:
        raise LayoutError("达到 maximum_rings，Campo 圆环生成未正常终止。")

    layout = GeneratedLayout("campo", tuple(rings))
    check = validate_layout(layout.coordinates, parameters)
    if not check.valid:
        raise LayoutError(check.reason or "Campo 布局不满足几何约束。")
    return layout


def validate_layout(
    coordinates: FloatArray,
    parameters: CommonParameters,
) -> GeometryCheck:
    """按题目口径检查尺寸、场地、禁区和严格中心距离约束。"""

    try:
        _validate_common_parameters(parameters)
    except LayoutError as exc:
        return GeometryCheck(False, str(exc), 0, math.inf, math.inf, math.inf)

    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return GeometryCheck(
            False,
            f"镜位坐标应为 N×2，实际形状为 {xy.shape}。",
            0,
            math.inf,
            math.inf,
            math.inf,
        )
    if xy.shape[0] == 0:
        return GeometryCheck(False, "镜场中没有定日镜。", 0, math.inf, 0.0, 0.0)
    if not np.all(np.isfinite(xy)):
        return GeometryCheck(
            False,
            "镜位坐标包含 NaN 或无穷值。",
            int(xy.shape[0]),
            math.inf,
            math.inf,
            math.inf,
        )

    field_radii = np.hypot(xy[:, 0], xy[:, 1])
    tower_distances = np.hypot(
        xy[:, 0] - parameters.tower_x,
        xy[:, 1] - parameters.tower_y,
    )
    maximum_field_radius = float(np.max(field_radii))
    minimum_tower_distance = float(np.min(tower_distances))
    if maximum_field_radius > parameters.field_radius + 1e-9:
        return GeometryCheck(
            False,
            "存在越过 350 m 场地边界的镜位。",
            int(xy.shape[0]),
            math.inf,
            maximum_field_radius,
            minimum_tower_distance,
        )
    if minimum_tower_distance < parameters.exclusion_radius - 1e-9:
        return GeometryCheck(
            False,
            "存在进入塔周 100 m 禁区的镜位。",
            int(xy.shape[0]),
            math.inf,
            maximum_field_radius,
            minimum_tower_distance,
        )

    if xy.shape[0] == 1:
        minimum_distance = math.inf
    else:
        distances, _ = cKDTree(xy).query(xy, k=2)
        minimum_distance = float(np.min(distances[:, 1]))
    if minimum_distance <= parameters.mirror_width + 5.0:
        return GeometryCheck(
            False,
            (
                "最小中心距离不满足严格约束："
                f"{minimum_distance:.9f} m ≤ "
                f"{parameters.mirror_width + 5.0:.9f} m。"
            ),
            int(xy.shape[0]),
            minimum_distance,
            maximum_field_radius,
            minimum_tower_distance,
        )

    return GeometryCheck(
        True,
        None,
        int(xy.shape[0]),
        minimum_distance,
        maximum_field_radius,
        minimum_tower_distance,
    )
