"""太阳锥光联合采样与有限高圆柱集热器求交。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from .config import SolverConfig
from .geometry import FloatArray, MirrorOrientation, PreparedField


def build_sobol_samples(sample_count: int, seed: int) -> FloatArray:
    """生成固定、可复现的四维 Sobol 样本。"""

    exponent = int(math.ceil(math.log2(sample_count)))
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    return sampler.random_base2(exponent)[:sample_count]


def _sun_disk_directions(
    sun_direction: FloatArray,
    samples: FloatArray,
    angular_radius: float,
) -> FloatArray:
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(sun_direction[2])) > 0.9
        else np.array([0.0, 0.0, 1.0])
    )
    tangent_one = np.cross(reference, sun_direction)
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(sun_direction, tangent_one)

    radial_angle = angular_radius * np.sqrt(samples[:, 2])
    polar_angle = 2.0 * math.pi * samples[:, 3]
    tangent = (
        np.cos(polar_angle)[:, None] * tangent_one[None, :]
        + np.sin(polar_angle)[:, None] * tangent_two[None, :]
    )
    directions = (
        np.cos(radial_angle)[:, None] * sun_direction[None, :]
        + np.sin(radial_angle)[:, None] * tangent
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def ray_cylinder_side_hits(
    origins: FloatArray,
    directions: FloatArray,
    tower_x: float,
    tower_y: float,
    radius: float,
    z_min: float,
    z_max: float,
    epsilon: float,
) -> NDArray[np.bool_]:
    """判断任意形状批量射线的两个正根中是否有有限圆柱侧面交点。"""

    origin_x = origins[..., 0] - tower_x
    origin_y = origins[..., 1] - tower_y
    direction_x = directions[..., 0]
    direction_y = directions[..., 1]

    a = direction_x**2 + direction_y**2
    b = 2.0 * (origin_x * direction_x + origin_y * direction_y)
    c = origin_x**2 + origin_y**2 - radius**2
    discriminant = b**2 - 4.0 * a * c

    valid = (a > epsilon) & (discriminant >= 0.0)
    square_root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = np.where(valid, 2.0 * a, 1.0)
    near = (-b - square_root) / denominator
    far = (-b + square_root) / denominator

    near_z = origins[..., 2] + near * directions[..., 2]
    far_z = origins[..., 2] + far * directions[..., 2]
    near_hit = (
        (near > epsilon)
        & (near_z >= z_min - epsilon)
        & (near_z <= z_max + epsilon)
    )
    far_hit = (
        (far > epsilon)
        & (far_z >= z_min - epsilon)
        & (far_z <= z_max + epsilon)
    )
    return valid & (near_hit | far_hit)


def calculate_truncation_efficiency(
    prepared: PreparedField,
    orientation: MirrorOrientation,
    sun_direction: FloatArray,
    solver: SolverConfig,
) -> FloatArray:
    """联合采样镜面位置和太阳圆盘方向，计算集热器截断效率。"""

    config = prepared.config
    samples = build_sobol_samples(solver.truncation_rays, solver.sobol_seed)
    local_width = (samples[:, 0] - 0.5) * config.mirror_width
    local_height = (samples[:, 1] - 0.5) * config.mirror_height
    sampled_sun = _sun_disk_directions(
        sun_direction,
        samples,
        config.solar_angular_radius_rad,
    )
    incoming = -sampled_sun

    efficiencies = np.empty(prepared.mirror_count, dtype=float)
    chunk_size = solver.truncation_chunk_size
    for start in range(0, prepared.mirror_count, chunk_size):
        stop = min(start + chunk_size, prepared.mirror_count)
        centers = prepared.centers[start:stop]
        normals = orientation.normals[start:stop]
        width_axes = orientation.width_axes[start:stop]
        height_axes = orientation.height_axes[start:stop]

        origins = (
            centers[:, None, :]
            + local_width[None, :, None] * width_axes[:, None, :]
            + local_height[None, :, None] * height_axes[:, None, :]
        )
        incoming_chunk = np.broadcast_to(
            incoming[None, :, :],
            origins.shape,
        )
        dot = np.einsum("csj,cj->cs", incoming_chunk, normals)
        reflected = (
            incoming_chunk - 2.0 * dot[:, :, None] * normals[:, None, :]
        )
        hits = ray_cylinder_side_hits(
            origins=origins,
            directions=reflected,
            tower_x=config.tower_x,
            tower_y=config.tower_y,
            radius=config.receiver_radius,
            z_min=config.receiver_z_min,
            z_max=config.receiver_z_max,
            epsilon=solver.ray_epsilon,
        )
        efficiencies[start:stop] = np.mean(hits, axis=1)

    return np.clip(efficiencies, 0.0, 1.0)
