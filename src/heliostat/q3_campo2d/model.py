"""动态 Campo 几何与径向—角度连续异构规格模型。"""

from __future__ import annotations

import json
import math
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..q2.layout import (
    CampoParameters,
    GeneratedLayout,
    LayoutRing,
    generate_campo_layout,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

RADIAL_NODE_COUNT = 5
ANGLE_TERM_COUNT = 2


@dataclass(frozen=True)
class Campo2DBase:
    """问题二 Campo 参数、正式圆环数和外层对称修剪标签。"""

    parameters: CampoParameters
    ring_count: int
    excluded_ring_angles: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Campo2DDesign:
    """Campo 几何、径向样条、角度修正和全局尺度的完整状态。"""

    tower_y: float
    initial_spacing: float
    spacing_growth: float
    ring_count: int
    size_nodes: tuple[float, ...]
    height_nodes: tuple[float, ...]
    size_angles: tuple[float, float]
    height_angles: tuple[float, float]
    area_scale: float = 1.0
    tower_x: float = 0.0

    def __post_init__(self) -> None:
        if len(self.size_nodes) != RADIAL_NODE_COUNT:
            raise ValueError("size_nodes 必须包含五个值。")
        if len(self.height_nodes) != RADIAL_NODE_COUNT:
            raise ValueError("height_nodes 必须包含五个值。")
        if len(self.size_angles) != ANGLE_TERM_COUNT:
            raise ValueError("size_angles 必须包含两个值。")
        if len(self.height_angles) != ANGLE_TERM_COUNT:
            raise ValueError("height_angles 必须包含两个值。")
        values = (
            self.tower_x,
            self.tower_y,
            self.initial_spacing,
            self.spacing_growth,
            *self.size_nodes,
            *self.height_nodes,
            *self.size_angles,
            *self.height_angles,
            self.area_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Campo2D 参数必须全部为有限数。")
        if self.ring_count < RADIAL_NODE_COUNT:
            raise ValueError("有效圆环数不能少于五个。")
        if self.initial_spacing <= 0.0:
            raise ValueError("初始径向行距必须大于 0。")
        if self.spacing_growth < 0.0:
            raise ValueError("径向行距增长量不能小于 0。")
        if self.area_scale <= 0.0:
            raise ValueError("全局尺度 lambda 必须大于 0。")

    @classmethod
    def uniform(
        cls,
        parameters: CampoParameters,
        *,
        ring_count: int,
    ) -> Campo2DDesign:
        return cls(
            tower_x=parameters.tower_x,
            tower_y=parameters.tower_y,
            initial_spacing=parameters.initial_spacing,
            spacing_growth=parameters.spacing_growth,
            ring_count=ring_count,
            size_nodes=(0.0,) * RADIAL_NODE_COUNT,
            height_nodes=(parameters.installation_height,) * RADIAL_NODE_COUNT,
            size_angles=(0.0, 0.0),
            height_angles=(0.0, 0.0),
            area_scale=1.0,
        )

    def canonical(self) -> Campo2DDesign:
        """消除全部尺寸节点共同平移产生的冗余。"""

        mean = float(np.mean(self.size_nodes))
        return replace(
            self,
            size_nodes=tuple(value - mean for value in self.size_nodes),
        )


@dataclass(frozen=True)
class Campo2DField:
    """由一组局部 Campo 几何参数确定的镜位与连续特征。"""

    parameters: CampoParameters
    layout: GeneratedLayout
    coordinates: FloatArray
    ring_indices: IntArray
    ring_member_indices: IntArray
    ring_radii: FloatArray
    zone_indices: IntArray
    nominal_ring_counts: IntArray
    actual_ring_counts: IntArray
    control_ring_indices: tuple[int, ...]
    control_radii: tuple[float, ...]
    radial_basis: FloatArray
    normalized_radii: FloatArray
    azimuth_angles: FloatArray
    angular_features: FloatArray

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def ring_count(self) -> int:
        return len(self.layout.rings)

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
class ExpandedSpecifications:
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    areas: FloatArray
    scales: FloatArray
    raw_size_shape: FloatArray
    centered_size_shape: FloatArray
    radial_size_component: FloatArray
    angular_size_component: FloatArray
    radial_height_component: FloatArray
    angular_height_component: FloatArray

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


def load_q2_campo_parameters(summary_path: str | Path) -> CampoParameters:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到问题二摘要：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("layout") != "campo":
        raise ValueError("问题二摘要的最终布局不是 Campo。")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("问题二摘要缺少 parameters。")
    return CampoParameters(**parameters)


def _read_coordinates(path: str | Path) -> FloatArray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到问题二正式镜位：{source}")
    rows: list[tuple[float, float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x_m", "y_m"} <= set(reader.fieldnames):
            raise ValueError("问题二镜位文件必须包含 x_m 和 y_m 列。")
        for row in reader:
            rows.append((float(row["x_m"]), float(row["y_m"])))
    if not rows:
        raise ValueError("问题二镜位文件为空。")
    return np.asarray(rows, dtype=float)


def load_q2_campo_base(
    summary_path: str | Path,
    coordinates_path: str | Path,
) -> Campo2DBase:
    """提取问题二正式前缀及其外层对称修剪结构标签。"""

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    parameters = load_q2_campo_parameters(summary_path)
    ring_count = int(summary.get("ring_count", 0))
    if ring_count < RADIAL_NODE_COUNT:
        raise ValueError("问题二摘要缺少合法 ring_count。")
    full = generate_campo_layout(parameters)
    if ring_count > len(full.rings):
        raise ValueError("问题二正式圆环数超过可重建的 Campo 圆环数。")
    prefix = GeneratedLayout("campo", full.rings[:ring_count])
    selected = _read_coordinates(coordinates_path)
    tree = cKDTree(prefix.coordinates)
    distances, indices = tree.query(selected, k=1)
    if np.any(distances > 1e-7) or np.unique(indices).size != indices.size:
        raise ValueError("问题二正式镜位无法唯一映射回 Campo 前缀。")
    missing = sorted(set(range(prefix.mirror_count)) - set(map(int, indices)))
    offsets = np.cumsum([0] + [ring.mirror_count for ring in prefix.rings])
    excluded: list[tuple[int, float]] = []
    for global_index in missing:
        ring_position = int(np.searchsorted(offsets, global_index, side="right") - 1)
        local_index = global_index - int(offsets[ring_position])
        ring = prefix.rings[ring_position]
        coordinate = ring.coordinates[local_index]
        theta = math.atan2(
            float(coordinate[0] - parameters.tower_x),
            float(coordinate[1] - parameters.tower_y),
        )
        excluded.append((ring_position + 1, theta))
    return Campo2DBase(
        parameters=parameters,
        ring_count=ring_count,
        excluded_ring_angles=tuple(excluded),
    )


def piecewise_linear_basis(
    radii: FloatArray,
    control_radii: tuple[float, ...],
) -> FloatArray:
    x = np.asarray(radii, dtype=float)
    knots = np.asarray(control_radii, dtype=float)
    if knots.shape != (RADIAL_NODE_COUNT,):
        raise ValueError("control_radii 必须包含五个节点。")
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("径向控制节点必须严格递增。")
    basis = np.zeros((x.size, RADIAL_NODE_COUNT), dtype=float)
    basis[x <= knots[0], 0] = 1.0
    basis[x >= knots[-1], -1] = 1.0
    for index in range(RADIAL_NODE_COUNT - 1):
        left = knots[index]
        right = knots[index + 1]
        active = (x >= left) & (x <= right)
        ratio = (x[active] - left) / (right - left)
        basis[active, index] = 1.0 - ratio
        basis[active, index + 1] = ratio
    if np.any(basis < -1e-12):
        raise RuntimeError("径向帽函数出现负值。")
    if not np.allclose(np.sum(basis, axis=1), 1.0, atol=1e-12):
        raise RuntimeError("径向帽函数不满足分割统一性。")
    return basis


def _nearest_unused_ring(
    target: int,
    *,
    ring_count: int,
    used: set[int],
) -> int:
    candidates = [ring for ring in range(1, ring_count + 1) if ring not in used]
    if not candidates:
        raise ValueError("无法得到五个互异的径向控制节点。")
    return min(candidates, key=lambda ring: (abs(ring - target), ring))


def _control_rings(layout: GeneratedLayout) -> tuple[int, ...]:
    ring_count = len(layout.rings)
    if ring_count < RADIAL_NODE_COUNT:
        raise ValueError("Campo 前缀不足五环，无法构造五节点样条。")
    first_zone = layout.rings[0].zone
    switch = next(
        (
            index
            for index, ring in enumerate(layout.rings, start=1)
            if ring.zone != first_zone
        ),
        1 + (ring_count - 1) // 2,
    )
    requested = (
        1,
        int(math.floor((1 + switch) / 2.0 + 0.5)),
        switch,
        int(math.floor((switch + ring_count) / 2.0 + 0.5)),
        ring_count,
    )
    selected: list[int] = []
    used: set[int] = set()
    for target in requested:
        chosen = (
            target
            if 1 <= target <= ring_count and target not in used
            else _nearest_unused_ring(target, ring_count=ring_count, used=used)
        )
        used.add(chosen)
        selected.append(chosen)
    return tuple(sorted(selected))


def _flatten_layout(
    layout: GeneratedLayout,
) -> tuple[FloatArray, IntArray, IntArray, FloatArray, IntArray, IntArray, IntArray]:
    coordinates: list[FloatArray] = []
    ring_indices: list[IntArray] = []
    member_indices: list[IntArray] = []
    radii: list[FloatArray] = []
    zones: list[IntArray] = []
    nominal_counts: list[IntArray] = []
    actual_counts: list[IntArray] = []
    for display_index, ring in enumerate(layout.rings, start=1):
        count = ring.mirror_count
        coordinates.append(ring.coordinates)
        ring_indices.append(np.full(count, display_index, dtype=np.int64))
        member_indices.append(np.arange(1, count + 1, dtype=np.int64))
        radii.append(np.full(count, ring.radius, dtype=float))
        zones.append(np.full(count, ring.zone, dtype=np.int64))
        nominal_counts.append(np.full(count, ring.nominal_count, dtype=np.int64))
        actual_counts.append(np.full(count, count, dtype=np.int64))
    return (
        np.concatenate(coordinates),
        np.concatenate(ring_indices),
        np.concatenate(member_indices),
        np.concatenate(radii),
        np.concatenate(zones),
        np.concatenate(nominal_counts),
        np.concatenate(actual_counts),
    )


def _angular_features(
    *,
    coordinates: FloatArray,
    ring_indices: IntArray,
    tower_x: float,
    tower_y: float,
) -> tuple[FloatArray, FloatArray]:
    theta = np.arctan2(
        coordinates[:, 0] - tower_x,
        coordinates[:, 1] - tower_y,
    )
    features = np.column_stack((np.cos(theta), np.cos(2.0 * theta)))
    for ring in np.unique(ring_indices):
        active = ring_indices == ring
        features[active] -= np.mean(features[active], axis=0)
    return np.asarray(theta, dtype=float), np.asarray(features, dtype=float)


def _apply_structural_exclusions(
    layout: GeneratedLayout,
    *,
    exclusions: tuple[tuple[int, float], ...],
    tower_x: float,
    tower_y: float,
) -> GeneratedLayout:
    rings: list[LayoutRing] = []
    for display_index, ring in enumerate(layout.rings, start=1):
        targets = [angle for index, angle in exclusions if index == display_index]
        if not targets:
            rings.append(ring)
            continue
        theta = np.arctan2(
            ring.coordinates[:, 0] - tower_x,
            ring.coordinates[:, 1] - tower_y,
        )
        available = set(range(ring.mirror_count))
        remove: set[int] = set()
        for target in targets:
            chosen = min(
                available,
                key=lambda index: abs(
                    math.atan2(
                        math.sin(float(theta[index]) - target),
                        math.cos(float(theta[index]) - target),
                    )
                ),
            )
            remove.add(chosen)
            available.remove(chosen)
        keep = np.asarray(
            [index not in remove for index in range(ring.mirror_count)],
            dtype=bool,
        )
        coordinates = np.asarray(ring.coordinates[keep], dtype=float)
        coordinates.setflags(write=False)
        rings.append(replace(ring, coordinates=coordinates))
    return GeneratedLayout(layout.kind, tuple(rings))


def build_campo_field(
    base: Campo2DBase,
    design: Campo2DDesign,
) -> Campo2DField:
    """按候选塔位和 Campo 参数重新生成并截取圆环前缀。"""

    canonical = design.canonical()
    parameters = replace(
        base.parameters,
        tower_x=canonical.tower_x,
        tower_y=canonical.tower_y,
        initial_spacing=canonical.initial_spacing,
        spacing_growth=canonical.spacing_growth,
    )
    full = generate_campo_layout(parameters)
    if canonical.ring_count > len(full.rings):
        raise ValueError(
            f"候选仅生成 {len(full.rings)} 个有效圆环，"
            f"不能保留前 {canonical.ring_count} 环。"
        )
    layout = GeneratedLayout("campo", full.rings[: canonical.ring_count])
    layout = _apply_structural_exclusions(
        layout,
        exclusions=base.excluded_ring_angles,
        tower_x=parameters.tower_x,
        tower_y=parameters.tower_y,
    )
    (
        coordinates,
        ring_indices,
        member_indices,
        radii,
        zones,
        nominal_counts,
        actual_counts,
    ) = _flatten_layout(layout)
    control_rings = _control_rings(layout)
    control_radii = tuple(layout.rings[index - 1].radius for index in control_rings)
    minimum_radius = float(np.min(radii))
    maximum_radius = float(np.max(radii))
    if maximum_radius <= minimum_radius:
        raise ValueError("Campo 径向范围不足，无法归一化半径。")
    normalized = (radii - minimum_radius) / (maximum_radius - minimum_radius)
    theta, angular = _angular_features(
        coordinates=coordinates,
        ring_indices=ring_indices,
        tower_x=parameters.tower_x,
        tower_y=parameters.tower_y,
    )
    return Campo2DField(
        parameters=parameters,
        layout=layout,
        coordinates=np.asarray(coordinates, dtype=float),
        ring_indices=np.asarray(ring_indices, dtype=np.int64),
        ring_member_indices=np.asarray(member_indices, dtype=np.int64),
        ring_radii=np.asarray(radii, dtype=float),
        zone_indices=np.asarray(zones, dtype=np.int64),
        nominal_ring_counts=np.asarray(nominal_counts, dtype=np.int64),
        actual_ring_counts=np.asarray(actual_counts, dtype=np.int64),
        control_ring_indices=control_rings,
        control_radii=tuple(float(value) for value in control_radii),
        radial_basis=piecewise_linear_basis(radii, control_radii),
        normalized_radii=np.asarray(normalized, dtype=float),
        azimuth_angles=theta,
        angular_features=angular,
    )


def expand_design(
    field: Campo2DField,
    design: Campo2DDesign,
) -> ExpandedSpecifications:
    canonical = design.canonical()
    radial_size = field.radial_basis @ np.asarray(canonical.size_nodes, dtype=float)
    angular_size = field.normalized_radii * (
        field.angular_features @ np.asarray(canonical.size_angles, dtype=float)
    )
    raw_size = radial_size + angular_size
    centered_size = raw_size - float(np.mean(raw_size))
    scales = canonical.area_scale * np.exp(centered_size)
    widths = field.base_width * scales
    heights = field.base_height * scales
    radial_height = field.radial_basis @ np.asarray(
        canonical.height_nodes,
        dtype=float,
    )
    angular_height = field.normalized_radii * (
        field.angular_features @ np.asarray(canonical.height_angles, dtype=float)
    )
    installation_heights = radial_height + angular_height
    return ExpandedSpecifications(
        widths=np.asarray(widths, dtype=float),
        heights=np.asarray(heights, dtype=float),
        installation_heights=np.asarray(installation_heights, dtype=float),
        areas=np.asarray(widths * heights, dtype=float),
        scales=np.asarray(scales, dtype=float),
        raw_size_shape=np.asarray(raw_size, dtype=float),
        centered_size_shape=np.asarray(centered_size, dtype=float),
        radial_size_component=np.asarray(radial_size, dtype=float),
        angular_size_component=np.asarray(angular_size, dtype=float),
        radial_height_component=np.asarray(radial_height, dtype=float),
        angular_height_component=np.asarray(angular_height, dtype=float),
    )


def validate_heterogeneous_field(
    *,
    field: Campo2DField,
    specifications: ExpandedSpecifications,
) -> HeterogeneousGeometryCheck:
    xy = field.coordinates
    widths = specifications.widths
    heights = specifications.heights
    center_zs = specifications.installation_heights
    count = field.mirror_count

    def invalid(reason: str) -> HeterogeneousGeometryCheck:
        return HeterogeneousGeometryCheck(
            valid=False,
            reason=reason,
            mirror_count=count,
            minimum_center_distance_m=math.inf,
            minimum_width_clearance_m=-math.inf,
            maximum_field_radius_m=math.inf,
            minimum_tower_distance_m=-math.inf,
            minimum_ground_clearance_m=-math.inf,
        )

    for name, values in (
        ("宽度", widths),
        ("高度", heights),
        ("安装高度", center_zs),
    ):
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            return invalid(f"{name}数组长度错误或包含非有限值。")
    if np.any(heights < 2.0) or np.any(heights > 8.0):
        return invalid("镜面高度必须位于 2 m 到 8 m。")
    if np.any(widths < 2.0) or np.any(widths > 8.0):
        return invalid("镜面宽度必须位于 2 m 到 8 m。")
    if np.any(widths < heights):
        return invalid("镜面宽度不能小于镜面高度。")
    if np.any(center_zs < 2.0) or np.any(center_zs > 6.0):
        return invalid("安装高度必须位于 2 m 到 6 m。")
    ground = center_zs - heights / 2.0
    if np.any(ground < -1e-12):
        return invalid("存在旋转时可能触地的镜面。")

    field_radii = np.hypot(xy[:, 0], xy[:, 1])
    tower_distances = np.hypot(
        xy[:, 0] - field.parameters.tower_x,
        xy[:, 1] - field.parameters.tower_y,
    )
    maximum_field_radius = float(np.max(field_radii))
    minimum_tower_distance = float(np.min(tower_distances))
    if maximum_field_radius > field.parameters.field_radius + 1e-9:
        return invalid("存在镜位超出圆形场地边界。")
    if minimum_tower_distance < field.parameters.exclusion_radius - 1e-9:
        return invalid("存在镜位进入塔周禁区。")

    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=2)
    minimum_center_distance = float(np.min(distances[:, 1]))
    maximum_reach = float(np.max(widths) + 5.0 + field.parameters.safety_epsilon)
    pairs = np.asarray(list(tree.query_pairs(maximum_reach)), dtype=np.int64)
    minimum_clearance = math.inf
    if pairs.size:
        pairs = pairs.reshape(-1, 2)
        pair_distances = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
        clearance = pair_distances - np.maximum(widths[pairs[:, 0]], widths[pairs[:, 1]]) - 5.0
        minimum_clearance = float(np.min(clearance))
        if minimum_clearance <= field.parameters.safety_epsilon - 1e-9:
            return invalid("存在镜对不满足异构宽度对应的中心距安全余量。")

    return HeterogeneousGeometryCheck(
        valid=True,
        reason=None,
        mirror_count=count,
        minimum_center_distance_m=minimum_center_distance,
        minimum_width_clearance_m=minimum_clearance,
        maximum_field_radius_m=maximum_field_radius,
        minimum_tower_distance_m=minimum_tower_distance,
        minimum_ground_clearance_m=float(np.min(ground)),
    )
