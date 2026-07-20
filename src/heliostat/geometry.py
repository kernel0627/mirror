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
    receiver_center: FloatArray
    receiver_directions: FloatArray
    receiver_distances: FloatArray
    atmospheric_efficiency: FloatArray

    @property
    def mirror_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def total_mirror_area(self) -> float:
        return self.mirror_count * self.config.mirror_area


@dataclass(frozen=True)
class MirrorOrientation:
    normals: FloatArray
    width_axes: FloatArray
    height_axes: FloatArray
    cosine_efficiency: FloatArray


def prepare_field(mirror_xy: FloatArray, config: FieldConfig) -> PreparedField:
    mirror_count = int(mirror_xy.shape[0])
    centers = np.column_stack(
        (
            mirror_xy,
            np.full(mirror_count, config.mirror_center_z, dtype=float),
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
