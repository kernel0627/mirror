"""镜面姿态、反射与基础几何计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import FieldConfig


FloatArray = NDArray[np.float64]


def normalize_rows(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError("不能归一化长度为零的向量。")
    return vectors / norms


def reflect(directions: FloatArray, normals: FloatArray) -> FloatArray:
    """根据 d - 2(d·n)n 计算反射方向。"""

    dot = np.sum(directions * normals, axis=-1, keepdims=True)
    return directions - 2.0 * dot * normals


@dataclass(frozen=True)
class PreparedField:
    config: FieldConfig
    centers: FloatArray
    mirror_widths: FloatArray
    mirror_heights: FloatArray
    mirror_areas: FloatArray
    receiver_center: FloatArray
    receiver_directions: FloatArray
    receiver_distances: FloatArray
    atmospheric_efficiency: FloatArray

    @property
    def mirror_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def total_mirror_area(self) -> float:
        return float(np.sum(self.mirror_areas))


@dataclass(frozen=True)
class MirrorOrientation:
    normals: FloatArray
    width_axes: FloatArray
    height_axes: FloatArray
    cosine_efficiency: FloatArray


def _per_mirror_values(
    values: FloatArray | None,
    *,
    fallback: float,
    mirror_count: int,
    name: str,
) -> FloatArray:
    if values is None:
        result = np.full(mirror_count, fallback, dtype=float)
    else:
        result = np.asarray(values, dtype=float)
        if result.ndim != 1 or result.shape[0] != mirror_count:
            raise ValueError(
                f"{name} 必须是一维且长度等于镜子数 {mirror_count}，"
                f"实际形状为 {result.shape}。"
            )
        result = result.copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 包含 NaN 或无穷值。")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} 必须全部大于 0。")
    return result


def prepare_field(
    mirror_xy: FloatArray,
    config: FieldConfig,
    *,
    mirror_widths: FloatArray | None = None,
    mirror_heights: FloatArray | None = None,
    mirror_center_zs: FloatArray | None = None,
) -> PreparedField:
    xy = np.asarray(mirror_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("镜位坐标必须为非空 N×2 数组。")
    if not np.all(np.isfinite(xy)):
        raise ValueError("镜位坐标包含 NaN 或无穷值。")

    mirror_count = int(xy.shape[0])
    widths = _per_mirror_values(
        mirror_widths,
        fallback=config.mirror_width,
        mirror_count=mirror_count,
        name="mirror_widths",
    )
    heights = _per_mirror_values(
        mirror_heights,
        fallback=config.mirror_height,
        mirror_count=mirror_count,
        name="mirror_heights",
    )
    center_zs = _per_mirror_values(
        mirror_center_zs,
        fallback=config.mirror_center_z,
        mirror_count=mirror_count,
        name="mirror_center_zs",
    )
    areas = widths * heights
    centers = np.column_stack(
        (
            xy,
            center_zs,
        )
    )
    receiver_center = np.array(
        [config.tower_x, config.tower_y, config.receiver_center_z],
        dtype=float,
    )
    receiver_vectors = receiver_center[None, :] - centers
    distances = np.linalg.norm(receiver_vectors, axis=1)
    receiver_directions = receiver_vectors / distances[:, None]
    atmospheric = (
        0.99321 - 0.0001176 * distances + 1.97e-8 * distances**2
    )
    atmospheric = np.clip(atmospheric, 0.0, 1.0)
    return PreparedField(
        config=config,
        centers=centers,
        mirror_widths=widths,
        mirror_heights=heights,
        mirror_areas=areas,
        receiver_center=receiver_center,
        receiver_directions=receiver_directions,
        receiver_distances=distances,
        atmospheric_efficiency=atmospheric,
    )


def calculate_orientation(
    prepared: PreparedField,
    sun_direction: FloatArray,
) -> MirrorOrientation:
    sun_rows = np.broadcast_to(sun_direction, prepared.receiver_directions.shape)
    normals = normalize_rows(sun_rows + prepared.receiver_directions)

    upward = np.broadcast_to(np.array([0.0, 0.0, 1.0]), normals.shape)
    width_axes = np.cross(upward, normals)
    weak = np.linalg.norm(width_axes, axis=1) < 1e-10
    width_axes[weak] = np.array([1.0, 0.0, 0.0])
    width_axes = normalize_rows(width_axes)
    height_axes = normalize_rows(np.cross(normals, width_axes))

    cosine = np.clip(normals @ sun_direction, 0.0, 1.0)
    return MirrorOrientation(
        normals=normals,
        width_axes=width_axes,
        height_axes=height_axes,
        cosine_efficiency=cosine,
    )


def maximum_reflection_error(
    prepared: PreparedField,
    orientation: MirrorOrientation,
    sun_direction: FloatArray,
) -> float:
    incoming = np.broadcast_to(-sun_direction, orientation.normals.shape)
    reflected = reflect(incoming, orientation.normals)
    errors = np.linalg.norm(reflected - prepared.receiver_directions, axis=1)
    return float(np.max(errors))
