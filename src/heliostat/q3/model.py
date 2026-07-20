"""第三问母场、六组规格展开和异构几何约束。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..q2.layout import (
    CampoParameters,
    GeneratedLayout,
    generate_campo_layout,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

GROUP_RING_RANGES = (
    (1, 1),
    (2, 5),
    (6, 11),
    (12, 14),
    (15, 20),
    (21, 28),
)
GROUP_COUNT = len(GROUP_RING_RANGES)
EXPECTED_GROUP_COUNTS = (72, 269, 283, 224, 357, 266)


@dataclass(frozen=True)
class CampoMotherField:
    """由问题二最终参数确定性重建的完整 1471 面 Campo 母场。"""

    parameters: CampoParameters
    layout: GeneratedLayout
    coordinates: FloatArray
    ring_indices: IntArray
    group_indices: IntArray

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def group_counts(self) -> tuple[int, ...]:
        return tuple(
            int(np.count_nonzero(self.group_indices == group))
            for group in range(GROUP_COUNT)
        )

    @property
    def base_width(self) -> float:
        return self.parameters.mirror_width

    @property
    def base_height(self) -> float:
        return self.parameters.mirror_height

    @property
    def base_installation_height(self) -> float:
        return self.parameters.installation_height


@dataclass(frozen=True)
class GroupDesign:
    """六组镜面尺度与安装高度。"""

    scales: tuple[float, ...]
    heights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scales) != GROUP_COUNT:
            raise ValueError(f"scales 必须包含 {GROUP_COUNT} 个值。")
        if len(self.heights) != GROUP_COUNT:
            raise ValueError(f"heights 必须包含 {GROUP_COUNT} 个值。")
        values = self.scales + self.heights
        if not all(math.isfinite(value) for value in values):
            raise ValueError("组设计参数必须全部为有限数。")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("组尺度必须全部大于 0。")

    @classmethod
    def uniform(cls, installation_height: float) -> GroupDesign:
        return cls(
            scales=(1.0,) * GROUP_COUNT,
            heights=(installation_height,) * GROUP_COUNT,
        )


@dataclass(frozen=True)
class ExpandedSpecifications:
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    areas: FloatArray

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.areas))


@dataclass(frozen=True)
class HeterogeneousGeometryCheck:
    valid: bool
    reason: str | None
    mirror_count: int
    minimum_center_distance_m: float
    minimum_width_clearance_m: float
    maximum_field_radius_m: float
    minimum_tower_distance_m: float
    minimum_ground_clearance_m: float


def _group_for_ring(ring_index: int) -> int:
    for group, (start, stop) in enumerate(GROUP_RING_RANGES):
        if start <= ring_index <= stop:
            return group
    raise ValueError(f"圆环 {ring_index} 不在第三问六组范围内。")


def load_q2_campo_parameters(
    summary_path: str | Path,
) -> CampoParameters:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到问题二摘要：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("layout") != "campo":
        raise ValueError("问题二摘要的最终布局不是 campo。")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("问题二摘要缺少 parameters。")
    return CampoParameters(**parameters)


def build_campo_mother_field(
    summary_path: str | Path,
    *,
    require_recorded_structure: bool = True,
) -> CampoMotherField:
    parameters = load_q2_campo_parameters(summary_path)
    layout = generate_campo_layout(parameters)
    if len(layout.rings) != 28:
        raise ValueError(
            f"第三问分组要求 28 个有效环，实际为 {len(layout.rings)}。"
        )

    coordinates: list[FloatArray] = []
    ring_indices: list[IntArray] = []
    group_indices: list[IntArray] = []
    for display_index, ring in enumerate(layout.rings, start=1):
        group = _group_for_ring(display_index)
        coordinates.append(ring.coordinates)
        ring_indices.append(
            np.full(ring.mirror_count, display_index, dtype=np.int64)
        )
        group_indices.append(
            np.full(ring.mirror_count, group, dtype=np.int64)
        )

    mother = CampoMotherField(
        parameters=parameters,
        layout=layout,
        coordinates=np.concatenate(coordinates, axis=0),
        ring_indices=np.concatenate(ring_indices),
        group_indices=np.concatenate(group_indices),
    )
    if (
        require_recorded_structure
        and mother.group_counts != EXPECTED_GROUP_COUNTS
    ):
        raise ValueError(
            "问题二 Campo 结构已变化："
            f"期望组镜数 {EXPECTED_GROUP_COUNTS}，"
            f"实际为 {mother.group_counts}。"
        )
    return mother


def expand_group_design(
    mother: CampoMotherField,
    design: GroupDesign,
) -> ExpandedSpecifications:
    group_indices = mother.group_indices
    scales = np.asarray(design.scales, dtype=float)[group_indices]
    installation_heights = np.asarray(
        design.heights,
        dtype=float,
    )[group_indices]
    widths = mother.base_width * scales
    heights = mother.base_height * scales
    return ExpandedSpecifications(
        widths=widths,
        heights=heights,
        installation_heights=installation_heights,
        areas=widths * heights,
    )


def individual_width_caps(
    coordinates: FloatArray,
    *,
    safety_epsilon: float = 0.01,
) -> FloatArray:
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("至少需要两个 N×2 镜位计算宽度上限。")
    distances = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return np.minimum(8.0, distances - 5.0 - safety_epsilon)


def group_width_caps(
    mother: CampoMotherField,
    *,
    safety_epsilon: float = 0.01,
) -> tuple[float, ...]:
    caps = individual_width_caps(
        mother.coordinates,
        safety_epsilon=safety_epsilon,
    )
    return tuple(
        float(np.min(caps[mother.group_indices == group]))
        for group in range(GROUP_COUNT)
    )


def validate_heterogeneous_field(
    *,
    coordinates: FloatArray,
    widths: FloatArray,
    heights: FloatArray,
    installation_heights: FloatArray,
    tower_x: float,
    tower_y: float,
    field_radius: float = 350.0,
    exclusion_radius: float = 100.0,
    safety_epsilon: float = 0.01,
) -> HeterogeneousGeometryCheck:
    xy = np.asarray(coordinates, dtype=float)
    mirror_widths = np.asarray(widths, dtype=float)
    mirror_heights = np.asarray(heights, dtype=float)
    center_zs = np.asarray(installation_heights, dtype=float)
    mirror_count = int(xy.shape[0]) if xy.ndim >= 1 else 0

    invalid = HeterogeneousGeometryCheck(
        valid=False,
        reason=None,
        mirror_count=mirror_count,
        minimum_center_distance_m=math.inf,
        minimum_width_clearance_m=-math.inf,
        maximum_field_radius_m=math.inf,
        minimum_tower_distance_m=-math.inf,
        minimum_ground_clearance_m=-math.inf,
    )
    if xy.ndim != 2 or xy.shape[1] != 2 or mirror_count == 0:
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "镜位必须为非空 N×2 数组。"}
        )
    for name, values in (
        ("宽度", mirror_widths),
        ("高度", mirror_heights),
        ("安装高度", center_zs),
    ):
        if values.ndim != 1 or values.shape[0] != mirror_count:
            return HeterogeneousGeometryCheck(
                **{
                    **invalid.__dict__,
                    "reason": f"{name}数组长度与镜子数不一致。",
                }
            )
    if not all(
        np.all(np.isfinite(values))
        for values in (xy, mirror_widths, mirror_heights, center_zs)
    ):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "几何数据包含 NaN 或无穷值。"}
        )
    if np.any(mirror_heights < 2.0) or np.any(mirror_heights > 8.0):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "镜面高度必须位于 2 m 到 8 m。"}
        )
    if np.any(mirror_widths < 2.0) or np.any(mirror_widths > 8.0):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "镜面宽度必须位于 2 m 到 8 m。"}
        )
    if np.any(mirror_widths < mirror_heights):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "镜面宽度不能小于镜面高度。"}
        )
    if np.any(center_zs < 2.0) or np.any(center_zs > 6.0):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "安装高度必须位于 2 m 到 6 m。"}
        )
    ground_clearance = center_zs - mirror_heights / 2.0
    if np.any(ground_clearance < -1e-12):
        return HeterogeneousGeometryCheck(
            **{**invalid.__dict__, "reason": "存在旋转时可能触地的镜面。"}
        )

    field_radii = np.hypot(xy[:, 0], xy[:, 1])
    tower_distances = np.hypot(
        xy[:, 0] - tower_x,
        xy[:, 1] - tower_y,
    )
    maximum_field_radius = float(np.max(field_radii))
    minimum_tower_distance = float(np.min(tower_distances))
    if maximum_field_radius > field_radius + 1e-9:
        return HeterogeneousGeometryCheck(
            **{
                **invalid.__dict__,
                "reason": "存在镜位超出圆形场地边界。",
                "maximum_field_radius_m": maximum_field_radius,
                "minimum_tower_distance_m": minimum_tower_distance,
                "minimum_ground_clearance_m": float(
                    np.min(ground_clearance)
                ),
            }
        )
    if minimum_tower_distance < exclusion_radius - 1e-9:
        return HeterogeneousGeometryCheck(
            **{
                **invalid.__dict__,
                "reason": "存在镜位进入塔周禁区。",
                "maximum_field_radius_m": maximum_field_radius,
                "minimum_tower_distance_m": minimum_tower_distance,
                "minimum_ground_clearance_m": float(
                    np.min(ground_clearance)
                ),
            }
        )

    tree = cKDTree(xy)
    nearest = tree.query(xy, k=2)[0][:, 1]
    minimum_center_distance = float(np.min(nearest))
    pairs = tree.query_pairs(
        r=13.0 + safety_epsilon + 1e-9,
        output_type="ndarray",
    )
    minimum_width_clearance = math.inf
    if pairs.size:
        deltas = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        distances = np.linalg.norm(deltas, axis=1)
        required = (
            np.maximum(
                mirror_widths[pairs[:, 0]],
                mirror_widths[pairs[:, 1]],
            )
            + 5.0
        )
        clearances = distances - required
        minimum_width_clearance = float(np.min(clearances))
        if minimum_width_clearance < safety_epsilon - 1e-9:
            return HeterogeneousGeometryCheck(
                valid=False,
                reason="存在镜对不满足异构宽度对应的中心距安全余量。",
                mirror_count=mirror_count,
                minimum_center_distance_m=minimum_center_distance,
                minimum_width_clearance_m=minimum_width_clearance,
                maximum_field_radius_m=maximum_field_radius,
                minimum_tower_distance_m=minimum_tower_distance,
                minimum_ground_clearance_m=float(
                    np.min(ground_clearance)
                ),
            )

    return HeterogeneousGeometryCheck(
        valid=True,
        reason=None,
        mirror_count=mirror_count,
        minimum_center_distance_m=minimum_center_distance,
        minimum_width_clearance_m=minimum_width_clearance,
        maximum_field_radius_m=maximum_field_radius,
        minimum_tower_distance_m=minimum_tower_distance,
        minimum_ground_clearance_m=float(np.min(ground_clearance)),
    )
