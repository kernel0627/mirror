"""题目参数与数值计算参数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FieldConfig:
    """第一问给定的镜场和环境参数。"""

    latitude_deg: float = 39.4
    altitude_km: float = 3.0
    field_radius: float = 350.0
    exclusion_radius: float = 100.0
    tower_x: float = 0.0
    tower_y: float = 0.0
    receiver_center_z: float = 86.0
    receiver_radius: float = 4.0
    receiver_height: float = 8.0
    mirror_width: float = 6.2
    mirror_height: float = 6.2
    mirror_center_z: float = 4.5
    reflectivity: float = 0.92
    solar_angular_radius_rad: float = 4.65e-3

    @property
    def receiver_z_min(self) -> float:
        return self.receiver_center_z - self.receiver_height / 2.0

    @property
    def receiver_z_max(self) -> float:
        return self.receiver_center_z + self.receiver_height / 2.0

    @property
    def mirror_area(self) -> float:
        return self.mirror_width * self.mirror_height

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SolverConfig:
    """可调的采样精度和加速参数。"""

    shadow_grid_size: int = 15
    truncation_rays: int = 256
    neighbor_radius_m: float = 60.0
    candidate_margin: float = 1.05
    ray_epsilon: float = 1e-7
    truncation_chunk_size: int = 128
    sobol_seed: int = 2023
    calculate_shadow: bool = True
    calculate_truncation: bool = True

    def __post_init__(self) -> None:
        if self.shadow_grid_size < 1:
            raise ValueError("shadow_grid_size 必须大于等于 1。")
        if self.truncation_rays < 1:
            raise ValueError("truncation_rays 必须大于等于 1。")
        if self.neighbor_radius_m <= 0.0:
            raise ValueError("neighbor_radius_m 必须大于 0。")
        if self.candidate_margin < 1.0:
            raise ValueError("candidate_margin 不能小于 1。")
        if self.ray_epsilon <= 0.0:
            raise ValueError("ray_epsilon 必须大于 0。")
        if self.truncation_chunk_size < 1:
            raise ValueError("truncation_chunk_size 必须大于等于 1。")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)
