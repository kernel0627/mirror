"""六组正式初值、21 维设计对象和逐镜规格展开。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from ..q2.layout import CampoParameters
from ._baseline import (
    GROUP_COUNT,
    CampoMotherField,
    ExpandedSpecifications,
    build_campo_mother_field,
)


TowerMode = Literal["A", "B"]


@dataclass(frozen=True)
class RefineDesign:
    """保留六区阶梯结构的完整候选。"""

    tower_mode: TowerMode
    tower_y: float
    initial_spacing: float
    spacing_growth: float
    widths: tuple[float, ...]
    mirror_heights: tuple[float, ...]
    installation_heights: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.tower_mode not in ("A", "B"):
            raise ValueError("tower_mode 必须为 A 或 B。")
        for name, values in (
            ("widths", self.widths),
            ("mirror_heights", self.mirror_heights),
            ("installation_heights", self.installation_heights),
        ):
            if len(values) != GROUP_COUNT:
                raise ValueError(f"{name} 必须包含六个值。")
        values = (
            self.tower_y,
            self.initial_spacing,
            self.spacing_growth,
            *self.widths,
            *self.mirror_heights,
            *self.installation_heights,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("候选参数必须全部为有限数。")

    def parameter(self, name: str) -> float:
        if name in ("tower_y", "initial_spacing", "spacing_growth"):
            return float(getattr(self, name))
        prefix = name[0]
        try:
            group = int(name[1:]) - 1
        except (ValueError, IndexError) as exc:
            raise KeyError(name) from exc
        if not 0 <= group < GROUP_COUNT:
            raise KeyError(name)
        values = {
            "w": self.widths,
            "h": self.mirror_heights,
            "H": self.installation_heights,
        }.get(prefix)
        if values is None:
            raise KeyError(name)
        return float(values[group])

    def with_parameter(self, name: str, value: float) -> RefineDesign:
        if name in ("tower_y", "initial_spacing", "spacing_growth"):
            return replace(self, **{name: float(value)})
        prefix = name[0]
        try:
            group = int(name[1:]) - 1
        except (ValueError, IndexError) as exc:
            raise KeyError(name) from exc
        attribute = {
            "w": "widths",
            "h": "mirror_heights",
            "H": "installation_heights",
        }.get(prefix)
        if attribute is None or not 0 <= group < GROUP_COUNT:
            raise KeyError(name)
        values = list(getattr(self, attribute))
        values[group] = float(value)
        return replace(self, **{attribute: tuple(values)})

    def to_dict(self) -> dict[str, object]:
        return {
            "tower_mode": self.tower_mode,
            "tower_x_m": 0.0,
            "tower_y_m": self.tower_y,
            "initial_spacing_m": self.initial_spacing,
            "spacing_growth_m_per_ring": self.spacing_growth,
            "widths_m": list(self.widths),
            "mirror_heights_m": list(self.mirror_heights),
            "installation_heights_m": list(self.installation_heights),
        }


@dataclass(frozen=True)
class RefineBaseline:
    mother: CampoMotherField
    design: RefineDesign
    expected_mirror_count: int
    expected_total_area_m2: float
    expected_power_mw: float
    expected_q_kw_m2: float
    expected_annual: dict[str, float]

    @property
    def parameters(self) -> CampoParameters:
        return self.mother.parameters


@dataclass(frozen=True)
class RefineField:
    """某一塔位语义和 Campo 参数下的前 28 个有效环。"""

    coordinates: np.ndarray
    ring_indices: np.ndarray
    group_indices: np.ndarray
    original_indices: np.ndarray
    mirror_set_hash: str
    outer_clipped_count: int
    geometry_center_y: float

    @property
    def mirror_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def group_counts(self) -> tuple[int, ...]:
        return tuple(
            int(np.count_nonzero(self.group_indices == group))
            for group in range(GROUP_COUNT)
        )


def load_baseline(
    *,
    q2_summary_path: str | Path,
    six_group_summary_path: str | Path,
) -> RefineBaseline:
    """从正式结果读取六组初值，禁止重新估计或手抄参数。"""

    mother = build_campo_mother_field(q2_summary_path)
    payload = json.loads(
        Path(six_group_summary_path).read_text(encoding="utf-8")
    )
    group_payload = payload.get("group_design", {}).get("groups")
    if not isinstance(group_payload, list) or len(group_payload) != GROUP_COUNT:
        raise ValueError("六组正式摘要缺少完整的 groups 数据。")
    ordered = sorted(group_payload, key=lambda item: int(item["group"]))
    design = RefineDesign(
        tower_mode="A",
        tower_y=float(payload["tower"]["y_m"]),
        initial_spacing=float(mother.parameters.initial_spacing),
        spacing_growth=float(mother.parameters.spacing_growth),
        widths=tuple(float(item["mirror_width_m"]) for item in ordered),
        mirror_heights=tuple(float(item["mirror_height_m"]) for item in ordered),
        installation_heights=tuple(
            float(item["installation_height_m"]) for item in ordered
        ),
    )
    annual = {key: float(value) for key, value in payload["annual"].items()}
    return RefineBaseline(
        mother=mother,
        design=design,
        expected_mirror_count=int(payload["mirror_count"]),
        expected_total_area_m2=float(payload["total_area_m2"]),
        expected_power_mw=float(annual["field_output_mw"]),
        expected_q_kw_m2=float(annual["unit_area_output_kw_m2"]),
        expected_annual=annual,
    )


def expand_specifications(
    field: RefineField,
    design: RefineDesign,
) -> ExpandedSpecifications:
    groups = field.group_indices
    widths = np.asarray(design.widths, dtype=float)[groups]
    heights = np.asarray(design.mirror_heights, dtype=float)[groups]
    installation = np.asarray(design.installation_heights, dtype=float)[groups]
    return ExpandedSpecifications(
        widths=np.asarray(widths, dtype=float),
        heights=np.asarray(heights, dtype=float),
        installation_heights=np.asarray(installation, dtype=float),
        areas=np.asarray(widths * heights, dtype=float),
    )
