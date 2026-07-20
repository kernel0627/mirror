"""太阳位置与 DNI 计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SolarState:
    month: int
    solar_time: float
    direction: FloatArray
    altitude_rad: float
    azimuth_rad: float
    declination_rad: float
    dni_kw_m2: float


MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def day_from_spring_equinox(month: int, day: int = 21) -> int:
    """以 3 月 21 日为第 0 天，返回题面赤纬公式所需的 D。"""

    if not 1 <= month <= 12:
        raise ValueError("month 必须位于 1 到 12。")
    if not 1 <= day <= MONTH_DAYS[month - 1]:
        raise ValueError("day 不在指定月份的有效范围内。")
    day_of_year = sum(MONTH_DAYS[: month - 1]) + day
    return day_of_year - 80


def calculate_solar_state(
    month: int,
    solar_time: float,
    latitude_deg: float,
    altitude_km: float,
) -> SolarState:
    """按题面附录计算东-北-天坐标下的太阳单位方向和 DNI。"""

    if not 0.0 <= solar_time <= 24.0:
        raise ValueError("solar_time 必须位于 0 到 24 小时。")

    d = day_from_spring_equinox(month)
    declination = math.asin(
        math.sin(2.0 * math.pi * d / 365.0)
        * math.sin(math.radians(23.45))
    )
    latitude = math.radians(latitude_deg)
    hour_angle = math.pi / 12.0 * (solar_time - 12.0)

    direction = np.array(
        [
            -math.cos(declination) * math.sin(hour_angle),
            math.cos(latitude) * math.sin(declination)
            - math.sin(latitude) * math.cos(declination) * math.cos(hour_angle),
            math.sin(latitude) * math.sin(declination)
            + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle),
        ],
        dtype=float,
    )
    direction /= np.linalg.norm(direction)

    altitude = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    azimuth = math.atan2(float(direction[0]), float(direction[1])) % (
        2.0 * math.pi
    )

    h = altitude_km
    a = 0.4237 - 0.00821 * (6.0 - h) ** 2
    b = 0.5055 + 0.00595 * (6.5 - h) ** 2
    c = 0.2711 + 0.01858 * (2.5 - h) ** 2
    if altitude <= 0.0:
        dni = 0.0
    else:
        dni = 1.366 * (a + b * math.exp(-c / math.sin(altitude)))

    return SolarState(
        month=month,
        solar_time=solar_time,
        direction=direction,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        declination_rad=declination,
        dni_kw_m2=dni,
    )
