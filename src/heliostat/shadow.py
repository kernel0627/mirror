"""规则网格射线追踪计算阴影遮挡效率。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .config import SolverConfig
from .geometry import FloatArray, MirrorOrientation, PreparedField


IntArray = NDArray[np.int64]


def mirror_grid_offsets(
    grid_size: int,
    mirror_width: float,
    mirror_height: float,
) -> FloatArray:
    """返回位于等面积小格中心的局部二维坐标。"""

    width_step = mirror_width / grid_size
    height_step = mirror_height / grid_size
    width_values = np.linspace(
        -mirror_width / 2.0 + width_step / 2.0,
        mirror_width / 2.0 - width_step / 2.0,
        grid_size,
    )
    height_values = np.linspace(
        -mirror_height / 2.0 + height_step / 2.0,
        mirror_height / 2.0 - height_step / 2.0,
        grid_size,
    )
    width_grid, height_grid = np.meshgrid(
        width_values,
        height_values,
        indexing="xy",
    )
    return np.column_stack((width_grid.ravel(), height_grid.ravel()))


def _direction_candidates(
    target_index: int,
    neighbors: IntArray,
    centers: FloatArray,
    direction: FloatArray,
    reach: float,
    maximum_distance: float | None = None,
) -> IntArray:
    if neighbors.size == 0:
        return neighbors

    delta = centers[neighbors] - centers[target_index]
    projection = delta @ direction
    perpendicular = delta - projection[:, None] * direction[None, :]
    perpendicular_distance = np.linalg.norm(perpendicular, axis=1)

    keep = (projection > -reach) & (perpendicular_distance <= reach)
    if maximum_distance is not None:
        keep &= projection < maximum_distance + reach
    return neighbors[keep]


def ray_rectangle_hits(
    origins: FloatArray,
    direction: FloatArray,
    rectangle_center: FloatArray,
    rectangle_normal: FloatArray,
    rectangle_width_axis: FloatArray,
    rectangle_height_axis: FloatArray,
    half_width: float,
    half_height: float,
    epsilon: float,
    maximum_distance: float | None = None,
) -> NDArray[np.bool_]:
    """判断一组同向射线是否与一面有限矩形相交。"""

    denominator = float(direction @ rectangle_normal)
    if abs(denominator) <= epsilon:
        return np.zeros(origins.shape[0], dtype=bool)

    distance = (rectangle_center - origins) @ rectangle_normal / denominator
    active = distance > epsilon
    if maximum_distance is not None:
        active &= distance < maximum_distance - epsilon
    if not np.any(active):
        return active

    intersections = origins + distance[:, None] * direction[None, :]
    relative = intersections - rectangle_center[None, :]
    local_width = relative @ rectangle_width_axis
    local_height = relative @ rectangle_height_axis
    active &= np.abs(local_width) <= half_width + epsilon
    active &= np.abs(local_height) <= half_height + epsilon
    return active


def _blocked_by_candidates(
    origins: FloatArray,
    direction: FloatArray,
    candidates: IntArray,
    prepared: PreparedField,
    orientation: MirrorOrientation,
    solver: SolverConfig,
    maximum_distance: float | None = None,
) -> NDArray[np.bool_]:
    blocked = np.zeros(origins.shape[0], dtype=bool)
    config = prepared.config

    for candidate in candidates:
        active_indices = np.flatnonzero(~blocked)
        if active_indices.size == 0:
            break
        hits = ray_rectangle_hits(
            origins=origins[active_indices],
            direction=direction,
            rectangle_center=prepared.centers[candidate],
            rectangle_normal=orientation.normals[candidate],
            rectangle_width_axis=orientation.width_axes[candidate],
            rectangle_height_axis=orientation.height_axes[candidate],
            half_width=config.mirror_width / 2.0,
            half_height=config.mirror_height / 2.0,
            epsilon=solver.ray_epsilon,
            maximum_distance=maximum_distance,
        )
        blocked[active_indices[hits]] = True

    return blocked


def calculate_shadow_blocking_efficiency(
    prepared: PreparedField,
    orientation: MirrorOrientation,
    sun_direction: FloatArray,
    solver: SolverConfig,
) -> FloatArray:
    """逐镜计算入射阴影和反射遮挡损失的采样点并集。"""

    mirror_count = prepared.mirror_count
    if mirror_count == 1:
        return np.ones(1, dtype=float)

    config = prepared.config
    offsets = mirror_grid_offsets(
        solver.shadow_grid_size,
        config.mirror_width,
        config.mirror_height,
    )
    sample_count = offsets.shape[0]
    tree = cKDTree(prepared.centers[:, :2])
    bounding_radius = 0.5 * np.hypot(
        config.mirror_width,
        config.mirror_height,
    )
    reach = 2.0 * bounding_radius * solver.candidate_margin
    efficiencies = np.empty(mirror_count, dtype=float)

    for index in range(mirror_count):
        points = (
            prepared.centers[index][None, :]
            + offsets[:, :1] * orientation.width_axes[index][None, :]
            + offsets[:, 1:] * orientation.height_axes[index][None, :]
        )
        neighbors = np.asarray(
            tree.query_ball_point(
                prepared.centers[index, :2],
                solver.neighbor_radius_m,
            ),
            dtype=np.int64,
        )
        neighbors = neighbors[neighbors != index]

        incoming_candidates = _direction_candidates(
            target_index=index,
            neighbors=neighbors,
            centers=prepared.centers,
            direction=sun_direction,
            reach=reach,
        )
        incoming_blocked = _blocked_by_candidates(
            origins=points,
            direction=sun_direction,
            candidates=incoming_candidates,
            prepared=prepared,
            orientation=orientation,
            solver=solver,
        )

        reflected_direction = prepared.receiver_directions[index]
        reflected_candidates = _direction_candidates(
            target_index=index,
            neighbors=neighbors,
            centers=prepared.centers,
            direction=reflected_direction,
            reach=reach,
            maximum_distance=prepared.receiver_distances[index],
        )
        reflected_blocked = _blocked_by_candidates(
            origins=points,
            direction=reflected_direction,
            candidates=reflected_candidates,
            prepared=prepared,
            orientation=orientation,
            solver=solver,
            maximum_distance=prepared.receiver_distances[index],
        )

        blocked = incoming_blocked | reflected_blocked
        efficiencies[index] = 1.0 - np.count_nonzero(blocked) / sample_count

    return np.clip(efficiencies, 0.0, 1.0)
