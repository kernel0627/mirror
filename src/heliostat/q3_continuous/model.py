"""固定问题二 Campo 镜位的五节点径向连续规格模型。"""

from __future__ import annotations

import csv
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

CONTROL_NODE_COUNT = 5
EXPECTED_RING_COUNT = 28
EXPECTED_FULL_MIRROR_COUNT = 1471
EXPECTED_Q2_MIRROR_COUNT = 1469


@dataclass(frozen=True)
class CampoMotherField:
    """问题二正式 1469 面镜场及其 Campo 径向样条特征。"""

    parameters: CampoParameters
    layout: GeneratedLayout
    coordinates: FloatArray
    ring_indices: IntArray
    ring_radii: FloatArray
    zone_indices: IntArray
    nominal_ring_counts: IntArray
    actual_ring_counts: IntArray
    original_indices: IntArray
    control_ring_indices: tuple[int, ...]
    control_radii: tuple[float, ...]
    radial_basis: FloatArray

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def ring_count(self) -> int:
        return int(np.unique(self.ring_indices).size)

    @property
    def base_width(self) -> float:
        return self.parameters.mirror_width

    @property
    def base_height(self) -> float:
        return self.parameters.mirror_height

    @property
    def base_installation_height(self) -> float:
        return self.parameters.installation_height

    @property
    def base_total_area_m2(self) -> float:
        return self.mirror_count * self.base_width * self.base_height


@dataclass(frozen=True)
class SplineDesign:
    """五个尺寸节点、五个高度节点和全局尺度系数。"""

    size_nodes: tuple[float, ...]
    height_nodes: tuple[float, ...]
    area_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.size_nodes) != CONTROL_NODE_COUNT:
            raise ValueError(
                f"size_nodes 必须包含 {CONTROL_NODE_COUNT} 个值。"
            )
        if len(self.height_nodes) != CONTROL_NODE_COUNT:
            raise ValueError(
                f"height_nodes 必须包含 {CONTROL_NODE_COUNT} 个值。"
            )
        values = self.size_nodes + self.height_nodes + (self.area_scale,)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("样条设计参数必须全部为有限数。")
        if self.area_scale <= 0.0:
            raise ValueError("全局面积尺度系数必须大于 0。")

    @classmethod
    def uniform(cls, installation_height: float) -> SplineDesign:
        return cls(
            size_nodes=(0.0,) * CONTROL_NODE_COUNT,
            height_nodes=(installation_height,) * CONTROL_NODE_COUNT,
            area_scale=1.0,
        )

    def canonical(self) -> SplineDesign:
        """消除给全部尺寸节点加同一常数产生的冗余。"""

        mean = float(np.mean(self.size_nodes))
        return SplineDesign(
            size_nodes=tuple(value - mean for value in self.size_nodes),
            height_nodes=self.height_nodes,
            area_scale=self.area_scale,
        )


@dataclass(frozen=True)
class ExpandedSpecifications:
    widths: FloatArray
    heights: FloatArray
    installation_heights: FloatArray
    areas: FloatArray
    scales: FloatArray
    raw_size_shape: FloatArray
    centered_size_shape: FloatArray

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


def _read_selected_coordinates(path: str | Path) -> FloatArray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到问题二最终镜位：{source}")
    rows: list[tuple[float, float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x_m", "y_m"} <= set(
            reader.fieldnames
        ):
            raise ValueError("问题二镜位文件必须包含 x_m 和 y_m 列。")
        for row in reader:
            rows.append((float(row["x_m"]), float(row["y_m"])))
    if not rows:
        raise ValueError("问题二镜位文件为空。")
    return np.asarray(rows, dtype=float)


def _selected_full_indices(
    full_coordinates: FloatArray,
    selected_coordinates: FloatArray,
    *,
    tolerance: float = 1e-7,
) -> IntArray:
    distances, indices = cKDTree(full_coordinates).query(
        selected_coordinates,
        k=1,
    )
    if np.any(distances > tolerance):
        raise ValueError(
            "问题二正式镜位无法映射回 Campo 结构，"
            f"最大坐标差为 {float(np.max(distances)):.3e} m。"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("问题二正式镜位映射出现重复 Campo 镜位。")
    return np.asarray(indices, dtype=np.int64)


def _actual_counts_by_ring(ring_indices: IntArray) -> IntArray:
    counts = np.empty(ring_indices.shape[0], dtype=np.int64)
    for ring in np.unique(ring_indices):
        active = ring_indices == ring
        counts[active] = int(np.count_nonzero(active))
    return counts


def _automatic_control_rings(
    *,
    ring_indices: IntArray,
    ring_radii: FloatArray,
    zone_indices: IntArray,
    nominal_counts: IntArray,
    actual_counts: IntArray,
) -> tuple[int, ...]:
    rings = np.unique(ring_indices)

    def ring_value(values: NDArray, ring: int) -> float:
        return float(values[np.flatnonzero(ring_indices == ring)[0]])

    first_ring = int(rings[0])
    outer_ring = int(rings[-1])
    first_crop = next(
        (
            int(ring)
            for ring in rings
            if ring_value(actual_counts, int(ring))
            < ring_value(nominal_counts, int(ring))
        ),
        first_ring,
    )
    first_zone = int(ring_value(zone_indices, first_ring))
    zone_switch = next(
        (
            int(ring)
            for ring in rings
            if int(ring_value(zone_indices, int(ring))) != first_zone
        ),
        outer_ring,
    )
    half_crop = min(
        (int(ring) for ring in rings),
        key=lambda ring: abs(
            ring_value(actual_counts, ring)
            / ring_value(nominal_counts, ring)
            - 0.5
        ),
    )
    selected = sorted(
        {first_ring, first_crop, zone_switch, half_crop, outer_ring}
    )
    while len(selected) < CONTROL_NODE_COUNT:
        gaps = [
            (
                ring_value(ring_radii, right)
                - ring_value(ring_radii, left),
                left,
                right,
            )
            for left, right in zip(selected[:-1], selected[1:])
        ]
        _, left, right = max(gaps)
        midpoint = 0.5 * (
            ring_value(ring_radii, left)
            + ring_value(ring_radii, right)
        )
        candidates = [
            int(ring)
            for ring in rings
            if left < int(ring) < right and int(ring) not in selected
        ]
        if not candidates:
            candidates = [
                int(ring)
                for ring in rings
                if int(ring) not in selected
            ]
        if not candidates:
            raise ValueError("无法补足五个互异的径向控制节点。")
        selected.append(
            min(
                candidates,
                key=lambda ring: abs(
                    ring_value(ring_radii, ring) - midpoint
                ),
            )
        )
        selected.sort()
    if len(selected) != CONTROL_NODE_COUNT:
        raise ValueError("自动控制节点数量不是 5。")
    return tuple(selected)


def piecewise_linear_basis(
    radii: FloatArray,
    control_radii: tuple[float, ...],
) -> FloatArray:
    """构造五节点分段线性帽函数，行和严格为 1。"""

    x = np.asarray(radii, dtype=float)
    knots = np.asarray(control_radii, dtype=float)
    if knots.shape != (CONTROL_NODE_COUNT,):
        raise ValueError("control_radii 必须包含五个节点。")
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("径向控制节点必须严格递增。")
    basis = np.zeros((x.size, CONTROL_NODE_COUNT), dtype=float)
    basis[x <= knots[0], 0] = 1.0
    basis[x >= knots[-1], -1] = 1.0
    for index in range(CONTROL_NODE_COUNT - 1):
        left = knots[index]
        right = knots[index + 1]
        active = (x >= left) & (x <= right)
        fraction = (x[active] - left) / (right - left)
        basis[active, index] = 1.0 - fraction
        basis[active, index + 1] = fraction
    if np.any(basis < -1e-12):
        raise RuntimeError("分段线性基函数出现负值。")
    if not np.allclose(np.sum(basis, axis=1), 1.0, atol=1e-12):
        raise RuntimeError("分段线性基函数的行和不为 1。")
    return basis


def build_campo_mother_field(
    summary_path: str | Path,
    *,
    selected_coordinates_path: str | Path,
    require_recorded_structure: bool = True,
) -> CampoMotherField:
    """固定问题二 1469 面镜位并自动生成五个径向控制节点。"""

    parameters = load_q2_campo_parameters(summary_path)
    layout = generate_campo_layout(parameters)
    if len(layout.rings) != EXPECTED_RING_COUNT:
        raise ValueError(
            f"期望 {EXPECTED_RING_COUNT} 个有效环，"
            f"实际为 {len(layout.rings)}。"
        )

    coordinates: list[FloatArray] = []
    ring_indices: list[IntArray] = []
    ring_radii: list[FloatArray] = []
    zone_indices: list[IntArray] = []
    nominal_counts: list[IntArray] = []
    for display_index, ring in enumerate(layout.rings, start=1):
        count = ring.mirror_count
        coordinates.append(ring.coordinates)
        ring_indices.append(
            np.full(count, display_index, dtype=np.int64)
        )
        ring_radii.append(np.full(count, ring.radius, dtype=float))
        zone_indices.append(np.full(count, ring.zone, dtype=np.int64))
        nominal_counts.append(
            np.full(count, ring.nominal_count, dtype=np.int64)
        )

    full_coordinates = np.concatenate(coordinates)
    selected_coordinates = _read_selected_coordinates(
        selected_coordinates_path
    )
    selected = _selected_full_indices(
        full_coordinates,
        selected_coordinates,
    )
    selected_rings = np.concatenate(ring_indices)[selected]
    selected_radii = np.concatenate(ring_radii)[selected]
    selected_zones = np.concatenate(zone_indices)[selected]
    selected_nominal = np.concatenate(nominal_counts)[selected]
    selected_actual = _actual_counts_by_ring(selected_rings)
    control_rings = _automatic_control_rings(
        ring_indices=selected_rings,
        ring_radii=selected_radii,
        zone_indices=selected_zones,
        nominal_counts=selected_nominal,
        actual_counts=selected_actual,
    )
    control_radii = tuple(
        float(selected_radii[np.flatnonzero(selected_rings == ring)[0]])
        for ring in control_rings
    )
    mother = CampoMotherField(
        parameters=parameters,
        layout=layout,
        coordinates=np.asarray(selected_coordinates, dtype=float),
        ring_indices=np.asarray(selected_rings, dtype=np.int64),
        ring_radii=np.asarray(selected_radii, dtype=float),
        zone_indices=np.asarray(selected_zones, dtype=np.int64),
        nominal_ring_counts=np.asarray(
            selected_nominal,
            dtype=np.int64,
        ),
        actual_ring_counts=np.asarray(selected_actual, dtype=np.int64),
        original_indices=np.asarray(selected, dtype=np.int64),
        control_ring_indices=control_rings,
        control_radii=control_radii,
        radial_basis=piecewise_linear_basis(
            selected_radii,
            control_radii,
        ),
    )
    if require_recorded_structure:
        if layout.mirror_count != EXPECTED_FULL_MIRROR_COUNT:
            raise ValueError(
                "完整 Campo 结构已变化："
                f"期望 {EXPECTED_FULL_MIRROR_COUNT} 面，"
                f"实际为 {layout.mirror_count} 面。"
            )
        if mother.mirror_count != EXPECTED_Q2_MIRROR_COUNT:
            raise ValueError(
                "问题二正式镜场已变化："
                f"期望 {EXPECTED_Q2_MIRROR_COUNT} 面，"
                f"实际为 {mother.mirror_count} 面。"
            )
        if mother.ring_count != EXPECTED_RING_COUNT:
            raise ValueError("问题二正式镜场没有保留全部 28 个环。")
    return mother


def expand_spline_design(
    mother: CampoMotherField,
    design: SplineDesign,
) -> ExpandedSpecifications:
    canonical = design.canonical()
    raw_shape = mother.radial_basis @ np.asarray(
        canonical.size_nodes,
        dtype=float,
    )
    centered_shape = raw_shape - float(np.mean(raw_shape))
    scales = canonical.area_scale * np.exp(centered_shape)
    widths = mother.base_width * scales
    heights = mother.base_height * scales
    installation_heights = mother.radial_basis @ np.asarray(
        canonical.height_nodes,
        dtype=float,
    )
    return ExpandedSpecifications(
        widths=widths,
        heights=heights,
        installation_heights=installation_heights,
        areas=widths * heights,
        scales=scales,
        raw_size_shape=raw_shape,
        centered_size_shape=centered_shape,
    )


def fit_spline_design(
    mother: CampoMotherField,
    *,
    widths: FloatArray,
    installation_heights: FloatArray,
) -> SplineDesign:
    """把已有逐镜连续规格最小二乘投影到新的五节点模型。"""

    mirror_widths = np.asarray(widths, dtype=float)
    center_heights = np.asarray(installation_heights, dtype=float)
    if mirror_widths.shape != (mother.mirror_count,):
        raise ValueError("拟合宽度数组长度与镜子数不一致。")
    if center_heights.shape != (mother.mirror_count,):
        raise ValueError("拟合安装高度数组长度与镜子数不一致。")
    log_scales = np.log(mirror_widths / mother.base_width)
    area_scale = math.exp(float(np.mean(log_scales)))
    centered = log_scales - math.log(area_scale)
    size_nodes = np.linalg.lstsq(
        mother.radial_basis,
        centered,
        rcond=None,
    )[0]
    height_nodes = np.linalg.lstsq(
        mother.radial_basis,
        center_heights,
        rcond=None,
    )[0]
    return SplineDesign(
        size_nodes=tuple(float(value) for value in size_nodes),
        height_nodes=tuple(float(value) for value in height_nodes),
        area_scale=area_scale,
    ).canonical()


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
