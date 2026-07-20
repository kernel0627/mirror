#!/usr/bin/env python3
"""
塔式太阳能定日镜场交互式三维模型。

这是 heliostat3DApp.m 的 Python 版本：

- PySide6 创建参数界面；
- PyVista/PyVistaQt 绘制可旋转的三维镜场；
- NumPy 计算太阳位置、同心环布局、镜面姿态和基础光学效率。

坐标约定：x 向东，y 向北，z 向上，单位均为米。

当前功率与原 MATLAB 程序一致，只计入余弦效率、大气透射率和镜面
反射率，不包含阴影遮挡效率与截断效率，因此仅作为三维展示中的
示意值，不能直接代替完整的数模评价结果。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


GUI_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import pyvista as pv
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    from pyvistaqt import QtInteractor

    GUI_AVAILABLE = True
except ModuleNotFoundError as exc:
    GUI_IMPORT_ERROR = exc
    GUI_AVAILABLE = False

    class _MissingGuiObject:
        """Allows the numerical model and --self-test to load without Qt."""

    QApplication = QCheckBox = QGroupBox = QHBoxLayout = QLabel = _MissingGuiObject
    QMainWindow = QPlainTextEdit = QPushButton = QScrollArea = _MissingGuiObject
    QSizePolicy = QSlider = QTabWidget = QVBoxLayout = QWidget = _MissingGuiObject
    QtInteractor = _MissingGuiObject
    Qt = QTimer = pv = None


FloatArray = NDArray[np.float64]


@dataclass
class HeliostatParameters:
    """All user-adjustable model parameters."""

    N: int = 500
    field_radius: float = 350.0
    tower_x: float = 0.0
    tower_y: float = 0.0
    exclusion_radius: float = 100.0
    layout_angle: float = 0.0

    mirror_width: float = 6.2
    mirror_height: float = 6.2
    mount_height: float = 4.5
    service_gap: float = 5.0

    receiver_z: float = 86.0
    receiver_diameter: float = 8.0
    receiver_height: float = 8.0

    latitude: float = 39.4
    month: int = 6
    solar_time: float = 12.0
    altitude_km: float = 3.0
    reflectivity: float = 0.92

    def copy(self) -> "HeliostatParameters":
        return HeliostatParameters(
            **{field.name: getattr(self, field.name) for field in fields(self)}
        )


@dataclass(frozen=True)
class ParameterSpec:
    group: str
    name: str
    label: str
    minimum: float
    maximum: float
    decimals: int


PARAMETER_SPECS = (
    ParameterSpec("镜场", "N", "定日镜数量 N", 20, 1500, 0),
    ParameterSpec("镜场", "field_radius", "镜场半径 (m)", 200, 500, 0),
    ParameterSpec("镜场", "tower_x", "塔位置 x_T (m)", -150, 150, 1),
    ParameterSpec("镜场", "tower_y", "塔位置 y_T (m)", -150, 150, 1),
    ParameterSpec("镜场", "exclusion_radius", "禁布半径 (m)", 50, 150, 0),
    ParameterSpec("镜场", "layout_angle", "布局旋转角 (deg)", 0, 360, 0),
    ParameterSpec("定日镜", "mirror_width", "镜面宽度 w (m)", 2, 8, 2),
    ParameterSpec("定日镜", "mirror_height", "镜面高度 h (m)", 2, 8, 2),
    ParameterSpec("定日镜", "mount_height", "安装高度 z (m)", 2, 6, 2),
    ParameterSpec("定日镜", "service_gap", "附加维护间距 (m)", 5, 20, 1),
    ParameterSpec("集热塔", "receiver_z", "集热器中心高度 (m)", 50, 150, 0),
    ParameterSpec("集热塔", "receiver_diameter", "集热器直径 (m)", 4, 16, 1),
    ParameterSpec("集热塔", "receiver_height", "集热器高度 (m)", 4, 16, 1),
    ParameterSpec("太阳/环境", "latitude", "纬度 (deg)", 10, 60, 1),
    ParameterSpec("太阳/环境", "month", "月份（每月21日）", 1, 12, 0),
    ParameterSpec("太阳/环境", "solar_time", "当地太阳时 (h)", 8, 16, 2),
    ParameterSpec("太阳/环境", "altitude_km", "海拔 H (km)", 0, 5, 2),
    ParameterSpec("太阳/环境", "reflectivity", "镜面反射率", 0.75, 0.98, 3),
)


@dataclass(frozen=True)
class SolarState:
    direction: FloatArray
    altitude: float
    azimuth: float
    declination: float
    dni: float


@dataclass(frozen=True)
class ModelState:
    solar: SolarState
    centres: FloatArray
    receiver_directions: FloatArray
    normals: FloatArray
    mirror_vertices: FloatArray
    mirror_faces: NDArray[np.int64]
    cosine_efficiency: FloatArray
    atmospheric_efficiency: FloatArray
    capacity: int
    total_area_m2: float
    indicative_power_mw: float
    mean_cosine_efficiency: float

    @property
    def placed_count(self) -> int:
        return int(self.centres.shape[0])


def normalize_rows(vectors: FloatArray) -> FloatArray:
    """Normalize a two-dimensional array row by row."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError("不能归一化长度为零的向量。")
    return vectors / norms


def enforce_constraints(
    parameters: HeliostatParameters,
    changed_name: str | None = None,
) -> HeliostatParameters:
    """Apply the same geometric constraints as the MATLAB application."""

    p = parameters

    # 2 <= mirror_height <= mirror_width <= 8
    if changed_name == "mirror_width" and p.mirror_height > p.mirror_width:
        p.mirror_height = p.mirror_width
    elif changed_name == "mirror_height" and p.mirror_height > p.mirror_width:
        p.mirror_width = p.mirror_height

    # Keep a small safety clearance beyond the theoretical h/2 requirement.
    clearance = 0.15
    minimum_mount_height = p.mirror_height / 2.0 + clearance
    if changed_name == "mount_height" and p.mount_height <= minimum_mount_height:
        p.mirror_height = max(
            2.0,
            min(p.mirror_width, 2.0 * (p.mount_height - clearance)),
        )
    elif p.mount_height <= minimum_mount_height:
        p.mount_height = min(6.0, minimum_mount_height)

    # The tower base must remain inside the field, with a 10 m rim.
    tower_radius = math.hypot(p.tower_x, p.tower_y)
    maximum_tower_radius = max(0.0, p.field_radius - 10.0)
    if tower_radius > maximum_tower_radius:
        scale = maximum_tower_radius / tower_radius
        p.tower_x *= scale
        p.tower_y *= scale

    p.N = int(round(p.N))
    p.month = int(round(p.month))
    return p


def solar_state(parameters: HeliostatParameters) -> SolarState:
    """Calculate Sun direction, angles and direct normal irradiance."""

    month_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    day_of_year = sum(month_days[: parameters.month - 1]) + 21
    days_from_spring_equinox = (day_of_year - 80) % 365

    declination = math.asin(
        math.sin(2.0 * math.pi * days_from_spring_equinox / 365.0)
        * math.sin(math.radians(23.45))
    )
    latitude = math.radians(parameters.latitude)
    hour_angle = math.pi / 12.0 * (parameters.solar_time - 12.0)

    # East-North-Up vector. atan2 later preserves morning/afternoon quadrant.
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

    h = parameters.altitude_km
    a = 0.4237 - 0.00821 * (6.0 - h) ** 2
    b = 0.5055 + 0.00595 * (6.5 - h) ** 2
    c = 0.2711 + 0.01858 * (2.5 - h) ** 2
    if altitude > 0.0:
        dni = 1.366 * (a + b * math.exp(-c / math.sin(altitude)))
    else:
        dni = 0.0

    return SolarState(direction, altitude, azimuth, declination, dni)


def make_concentric_field(
    parameters: HeliostatParameters,
) -> tuple[FloatArray, int]:
    """
    Generate tower-centred concentric rings.

    Radial spacing and same-ring chord length are both at least
    mirror_width + service_gap. A mirror is accepted only when its centre
    leaves enough room for the mirror half diagonal inside the field rim.
    """

    spacing = parameters.mirror_width + parameters.service_gap
    half_diagonal = 0.5 * math.hypot(
        parameters.mirror_width,
        parameters.mirror_height,
    )
    rim = half_diagonal + 0.5
    first_radius = parameters.exclusion_radius + rim
    last_radius = parameters.field_radius + math.hypot(
        parameters.tower_x,
        parameters.tower_y,
    )

    if first_radius > last_radius:
        return np.empty((0, 2), dtype=float), 0

    ring_count = int(math.floor((last_radius - first_radius) / spacing)) + 1
    rings = first_radius + spacing * np.arange(ring_count, dtype=float)
    base_angle = math.radians(parameters.layout_angle)
    accepted_rings: list[FloatArray] = []

    for ring_index, radius in enumerate(rings, start=1):
        ratio = min(1.0, spacing / (2.0 * radius))
        maximum_count = max(1, int(math.floor(math.pi / math.asin(ratio))))
        angles = (
            base_angle
            + np.arange(maximum_count, dtype=float)
            * (2.0 * math.pi / maximum_count)
            + (ring_index % 2) * math.pi / maximum_count
        )
        ring_points = np.column_stack(
            (
                parameters.tower_x + radius * np.cos(angles),
                parameters.tower_y + radius * np.sin(angles),
            )
        )
        inside = np.linalg.norm(ring_points, axis=1) <= (
            parameters.field_radius - rim
        )
        if np.any(inside):
            accepted_rings.append(ring_points[inside])

    if not accepted_rings:
        return np.empty((0, 2), dtype=float), 0

    all_points = np.vstack(accepted_rings)
    capacity = int(all_points.shape[0])
    return all_points[: min(parameters.N, capacity)].copy(), capacity


def build_mirror_mesh(
    centres: FloatArray,
    normals: FloatArray,
    width: float,
    height: float,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Build four vertices and one quadrilateral face for every mirror."""

    count = int(centres.shape[0])
    if count == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=np.int64)

    upward = np.broadcast_to(np.array([0.0, 0.0, 1.0]), normals.shape)
    width_axes = np.cross(upward, normals)
    weak = np.linalg.norm(width_axes, axis=1) < 1e-9
    width_axes[weak] = np.array([1.0, 0.0, 0.0])
    width_axes = normalize_rows(width_axes)

    height_axes = normalize_rows(np.cross(normals, width_axes))
    signs = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    vertices = (
        centres[:, None, :]
        + signs[None, :, 0, None] * (width / 2.0) * width_axes[:, None, :]
        + signs[None, :, 1, None] * (height / 2.0) * height_axes[:, None, :]
    ).reshape((-1, 3))
    faces = np.arange(4 * count, dtype=np.int64).reshape((count, 4))
    return vertices, faces


def calculate_model(parameters: HeliostatParameters) -> ModelState:
    """Run all calculations needed by one screen refresh."""

    sun = solar_state(parameters)
    xy, capacity = make_concentric_field(parameters)
    count = int(xy.shape[0])

    if count == 0:
        empty_vectors = np.empty((0, 3), dtype=float)
        empty_scalars = np.empty(0, dtype=float)
        return ModelState(
            solar=sun,
            centres=empty_vectors,
            receiver_directions=empty_vectors.copy(),
            normals=empty_vectors.copy(),
            mirror_vertices=empty_vectors.copy(),
            mirror_faces=np.empty((0, 4), dtype=np.int64),
            cosine_efficiency=empty_scalars,
            atmospheric_efficiency=empty_scalars.copy(),
            capacity=capacity,
            total_area_m2=0.0,
            indicative_power_mw=0.0,
            mean_cosine_efficiency=float("nan"),
        )

    centres = np.column_stack(
        (xy, np.full(count, parameters.mount_height, dtype=float))
    )
    target = np.array(
        [parameters.tower_x, parameters.tower_y, parameters.receiver_z],
        dtype=float,
    )
    receiver_directions = normalize_rows(target - centres)
    normals = normalize_rows(receiver_directions + sun.direction[None, :])
    cosine_efficiency = np.clip(
        np.einsum("ij,j->i", normals, sun.direction),
        0.0,
        1.0,
    )

    vertices, faces_array = build_mirror_mesh(
        centres,
        normals,
        parameters.mirror_width,
        parameters.mirror_height,
    )

    receiver_distance = np.linalg.norm(target - centres, axis=1)
    atmospheric_efficiency = (
        0.99321
        - 0.0001176 * receiver_distance
        + 1.97e-8 * receiver_distance**2
    )
    atmospheric_efficiency = np.clip(atmospheric_efficiency, 0.0, 1.0)

    mirror_area = parameters.mirror_width * parameters.mirror_height
    total_area = count * mirror_area
    indicative_power_mw = (
        sun.dni
        * float(
            np.sum(
                mirror_area
                * cosine_efficiency
                * atmospheric_efficiency
                * parameters.reflectivity
            )
        )
        / 1000.0
    )

    return ModelState(
        solar=sun,
        centres=centres,
        receiver_directions=receiver_directions,
        normals=normals,
        mirror_vertices=vertices,
        mirror_faces=faces_array,
        cosine_efficiency=cosine_efficiency,
        atmospheric_efficiency=atmospheric_efficiency,
        capacity=capacity,
        total_area_m2=total_area,
        indicative_power_mw=indicative_power_mw,
        mean_cosine_efficiency=float(np.mean(cosine_efficiency)),
    )


def format_status(
    parameters: HeliostatParameters,
    state: ModelState,
) -> str:
    warning = ""
    if state.placed_count < parameters.N:
        warning = (
            "\n警告：当前约束最多生成 "
            f"{state.capacity} 面，少于请求的 {parameters.N} 面。"
        )

    return "\n".join(
        (
            f"已布置/请求：{state.placed_count} / {parameters.N} 面",
            (
                "最小中心距："
                f"{parameters.mirror_width + parameters.service_gap:.2f} m "
                f"(= w + {parameters.service_gap:.1f})"
            ),
            f"总镜面面积：{state.total_area_m2:.1f} m²",
            f"太阳高度角：{math.degrees(state.solar.altitude):.2f} deg",
            (
                "太阳方位角："
                f"{math.degrees(state.solar.azimuth):.2f} deg（北起顺时针）"
            ),
            f"太阳赤纬角：{math.degrees(state.solar.declination):.2f} deg",
            f"DNI：{state.solar.dni:.3f} kW/m²",
            f"平均余弦效率：{state.mean_cosine_efficiency:.4f}",
            (
                f"示意热功率：{state.indicative_power_mw:.3f} MW"
                f"（未计遮挡与截断）{warning}"
            ),
        )
    )


def _packed_faces(faces_array: NDArray[np.int64]) -> NDArray[np.int64]:
    counts = np.full((faces_array.shape[0], 1), faces_array.shape[1], dtype=np.int64)
    return np.hstack((counts, faces_array)).ravel()


def _line_mesh(segments: FloatArray) -> Any:
    """Create a PyVista mesh from an (n, 2, 3) segment array."""

    points = np.asarray(segments, dtype=float).reshape((-1, 3))
    indices = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 2))
    lines = np.hstack(
        (np.full((indices.shape[0], 1), 2, dtype=np.int64), indices)
    ).ravel()
    return pv.PolyData(points, lines=lines)


def _polyline_mesh(points: FloatArray) -> Any:
    """Create one connected PyVista polyline."""

    line = np.concatenate(
        (
            np.array([points.shape[0]], dtype=np.int64),
            np.arange(points.shape[0], dtype=np.int64),
        )
    )
    return pv.PolyData(points, lines=line)


def _frustum_mesh(
    x: float,
    y: float,
    height: float,
    lower_radius: float = 4.0,
    upper_radius: float = 2.3,
    resolution: int = 48,
) -> Any:
    """Create the tapered tower body used by the MATLAB model."""

    angles = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
    lower = np.column_stack(
        (
            x + lower_radius * np.cos(angles),
            y + lower_radius * np.sin(angles),
            np.zeros(resolution),
        )
    )
    upper = np.column_stack(
        (
            x + upper_radius * np.cos(angles),
            y + upper_radius * np.sin(angles),
            np.full(resolution, height),
        )
    )
    points = np.vstack((lower, upper))
    current = np.arange(resolution, dtype=np.int64)
    following = (current + 1) % resolution
    faces_array = np.column_stack(
        (
            current,
            following,
            resolution + following,
            resolution + current,
        )
    )
    return pv.PolyData(points, faces=_packed_faces(faces_array))


class ScaledSlider(QWidget):
    """Integer Qt slider exposing a formatted floating-point value."""

    def __init__(
        self,
        spec: ParameterSpec,
        value: float,
        callback: Callable[[str, float], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.scale = 10**spec.decimals
        self.callback = callback

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(8)

        name_label = QLabel(spec.label)
        name_label.setMinimumWidth(148)
        name_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            int(round(spec.minimum * self.scale)),
            int(round(spec.maximum * self.scale)),
        )
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, int((spec.maximum - spec.minimum) * self.scale / 20)))
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.value_label = QLabel()
        self.value_label.setMinimumWidth(62)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setStyleSheet("font-weight: 600;")

        layout.addWidget(name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_integer_value_changed)
        self.set_value(value)

    def _format_value(self, value: float) -> str:
        return f"{value:.{self.spec.decimals}f}"

    def _on_integer_value_changed(self, integer_value: int) -> None:
        value = integer_value / self.scale
        self.value_label.setText(self._format_value(value))
        self.callback(self.spec.name, value)

    def set_value(self, value: float) -> None:
        integer_value = int(round(value * self.scale))
        previous = self.slider.blockSignals(True)
        self.slider.setValue(integer_value)
        self.slider.blockSignals(previous)
        self.value_label.setText(self._format_value(value))


class Heliostat3DApp(QMainWindow):
    """Main PySide6 window."""

    def __init__(self) -> None:
        super().__init__()
        self.defaults = HeliostatParameters()
        self.parameters = self.defaults.copy()
        self.parameter_widgets: dict[str, ScaledSlider] = {}
        self._has_rendered = False

        self.setWindowTitle("定日镜场交互式三维模型")
        self.resize(1500, 900)
        self.setMinimumSize(1120, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        plot_group = QGroupBox("三维模型（鼠标可旋转、缩放和平移）")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(4, 6, 4, 4)
        self.plotter = QtInteractor(plot_group)
        self.plotter.set_background("#f4f6fa")
        plot_layout.addWidget(self.plotter.interactor)
        main_layout.addWidget(plot_group, 1)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(70)
        self.render_timer.timeout.connect(self.render_model)

        self.render_model()

    def _create_control_panel(self) -> QGroupBox:
        panel = QGroupBox("参数控制")
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(430)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(7, 9, 7, 7)

        tabs = QTabWidget()
        for group_name in ("镜场", "定日镜", "集热塔", "太阳/环境"):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(1, 5, 1, 5)
            page_layout.setSpacing(1)

            for spec in PARAMETER_SPECS:
                if spec.group != group_name:
                    continue
                widget = ScaledSlider(
                    spec,
                    float(getattr(self.parameters, spec.name)),
                    self.parameter_changed,
                )
                self.parameter_widgets[spec.name] = widget
                page_layout.addWidget(widget)

            page_layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            tabs.addTab(scroll, group_name)
        layout.addWidget(tabs, 1)

        button_row = QHBoxLayout()
        reset_parameters_button = QPushButton("恢复默认")
        reset_parameters_button.clicked.connect(self.reset_parameters)
        reset_view_button = QPushButton("复位视角")
        reset_view_button.clicked.connect(self.reset_view)
        self.ray_checkbox = QCheckBox("显示光路")
        self.ray_checkbox.setChecked(True)
        self.ray_checkbox.toggled.connect(lambda _: self.schedule_render())
        button_row.addWidget(reset_parameters_button)
        button_row.addWidget(reset_view_button)
        button_row.addWidget(self.ray_checkbox)
        layout.addLayout(button_row)

        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(190)
        self.status.setMaximumHeight(220)
        self.status.setStyleSheet(
            "QPlainTextEdit {"
            "background: #fbfcff;"
            "font-family: Menlo, Monaco, monospace;"
            "font-size: 12px;"
            "}"
        )
        layout.addWidget(self.status)
        return panel

    def parameter_changed(self, name: str, value: float) -> None:
        if name in {"N", "month"}:
            setattr(self.parameters, name, int(round(value)))
        else:
            setattr(self.parameters, name, float(value))
        enforce_constraints(self.parameters, name)
        self.sync_controls()
        self.schedule_render()

    def sync_controls(self) -> None:
        for name, widget in self.parameter_widgets.items():
            widget.set_value(float(getattr(self.parameters, name)))

    def schedule_render(self) -> None:
        self.status.setPlainText("正在重新计算并绘制……")
        self.render_timer.start()

    def reset_parameters(self) -> None:
        self.parameters = self.defaults.copy()
        self.sync_controls()
        self.schedule_render()

    def reset_view(self) -> None:
        p = self.parameters
        limit = p.field_radius * 1.08
        z_max = max(p.receiver_z + p.receiver_height / 2.0 + 110.0, 190.0)
        focus = np.array([p.tower_x, p.tower_y, z_max * 0.24])
        azimuth = math.radians(42.0)
        elevation = math.radians(28.0)
        distance = 2.35 * max(limit, z_max)
        offset = distance * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        self.plotter.camera_position = [
            (focus + offset).tolist(),
            focus.tolist(),
            [0.0, 0.0, 1.0],
        ]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def render_model(self) -> None:
        state = calculate_model(self.parameters)
        camera_position = self.plotter.camera_position if self._has_rendered else None

        self.plotter.clear()
        self.plotter.set_background("#f4f6fa")
        self._draw_ground()
        self._draw_tower()

        if state.placed_count:
            mirror_mesh = pv.PolyData(
                state.mirror_vertices,
                faces=_packed_faces(state.mirror_faces),
            )
            mirror_mesh.cell_data["eta_cos"] = state.cosine_efficiency
            self.plotter.add_mesh(
                mirror_mesh,
                scalars="eta_cos",
                preference="cell",
                cmap="turbo",
                clim=(0.5, 1.0),
                show_edges=True,
                edge_color="#142932",
                line_width=0.35,
                opacity=0.96,
                scalar_bar_args={
                    "title": "eta_cos",
                    "n_labels": 6,
                    "fmt": "%.2f",
                    "vertical": True,
                },
            )
            self._draw_support_posts(state.centres)

            if self.ray_checkbox.isChecked():
                self._draw_representative_rays(state)

        self._draw_sun_arrow(state.solar.direction)

        p = self.parameters
        limit = p.field_radius * 1.08
        z_max = max(p.receiver_z + p.receiver_height / 2.0 + 110.0, 190.0)
        bounds = (-limit, limit, -limit, limit, 0.0, z_max)
        self.plotter.show_grid(
            bounds=bounds,
            xtitle="x / East (m)",
            ytitle="y / North (m)",
            ztitle="z / Up (m)",
            color="#4b5563",
            font_size=10,
        )
        self.plotter.add_text(
            f"{p.month} 月 21 日  {p.solar_time:05.2f} 当地太阳时",
            position="upper_left",
            font_size=12,
            color="#1f2937",
        )

        if camera_position is None:
            self.reset_view()
        else:
            self.plotter.camera_position = camera_position
            self.plotter.reset_camera_clipping_range()
            self.plotter.render()

        self.status.setPlainText(format_status(p, state))
        self._has_rendered = True

    def _draw_ground(self) -> None:
        p = self.parameters
        ground = pv.Disc(
            center=(0.0, 0.0, 0.0),
            inner=0.0,
            outer=p.field_radius,
            normal=(0.0, 0.0, 1.0),
            r_res=1,
            c_res=240,
        )
        self.plotter.add_mesh(
            ground,
            color="#d8cc9f",
            opacity=0.38,
            show_edges=False,
        )

        angles = np.linspace(0.0, 2.0 * math.pi, 241)
        field_boundary = np.column_stack(
            (
                p.field_radius * np.cos(angles),
                p.field_radius * np.sin(angles),
                np.full(angles.size, 0.05),
            )
        )
        self.plotter.add_mesh(
            _polyline_mesh(field_boundary),
            color="#385238",
            line_width=3.0,
        )

        exclusion_boundary = np.column_stack(
            (
                p.tower_x + p.exclusion_radius * np.cos(angles),
                p.tower_y + p.exclusion_radius * np.sin(angles),
                np.full(angles.size, 0.12),
            )
        )
        self.plotter.add_mesh(
            _polyline_mesh(exclusion_boundary),
            color="#d3392f",
            line_width=2.5,
        )

    def _draw_tower(self) -> None:
        p = self.parameters
        lower_receiver = p.receiver_z - p.receiver_height / 2.0
        tower = _frustum_mesh(
            p.tower_x,
            p.tower_y,
            lower_receiver,
        )
        self.plotter.add_mesh(
            tower,
            color="#aeb2b9",
            smooth_shading=True,
        )

        receiver = pv.Cylinder(
            center=(p.tower_x, p.tower_y, p.receiver_z),
            direction=(0.0, 0.0, 1.0),
            radius=p.receiver_diameter / 2.0,
            height=p.receiver_height,
            resolution=64,
            capping=True,
        )
        self.plotter.add_mesh(
            receiver,
            color="#e94c13",
            smooth_shading=True,
            show_edges=False,
        )

    def _draw_support_posts(self, centres: FloatArray) -> None:
        bases = centres.copy()
        bases[:, 2] = 0.0
        segments = np.stack((bases, centres), axis=1)
        self.plotter.add_mesh(
            _line_mesh(segments),
            color="#44484d",
            line_width=1.0,
        )

    def _draw_representative_rays(self, state: ModelState) -> None:
        count = state.placed_count
        take = np.unique(
            np.rint(np.linspace(0, count - 1, min(14, count))).astype(int)
        )
        centres = state.centres[take]
        sun = state.solar.direction
        incoming_starts = centres + 38.0 * sun[None, :]
        incoming_segments = np.stack((incoming_starts, centres), axis=1)
        self.plotter.add_mesh(
            _line_mesh(incoming_segments),
            color="#ffb80a",
            line_width=1.2,
        )

        target = np.array(
            [
                self.parameters.tower_x,
                self.parameters.tower_y,
                self.parameters.receiver_z,
            ]
        )
        targets = np.broadcast_to(target, centres.shape)
        reflected_segments = np.stack((centres, targets), axis=1)
        self.plotter.add_mesh(
            _line_mesh(reflected_segments),
            color="#f04424",
            line_width=1.1,
        )

    def _draw_sun_arrow(self, sun_direction: FloatArray) -> None:
        p = self.parameters
        base = np.array(
            [
                p.tower_x,
                p.tower_y,
                max(p.receiver_z + 18.0, 110.0),
            ]
        )
        scale = 90.0
        arrow = pv.Arrow(
            start=base,
            direction=sun_direction,
            tip_length=0.18,
            tip_radius=0.06,
            shaft_radius=0.018,
            scale=scale,
        )
        self.plotter.add_mesh(arrow, color="#ff8a00")

    def closeEvent(self, event: Any) -> None:
        self.plotter.close()
        super().closeEvent(event)


def run_self_test() -> None:
    """Exercise the numerical model without opening the desktop window."""

    parameters = HeliostatParameters()
    state = calculate_model(parameters)

    assert np.isclose(np.linalg.norm(state.solar.direction), 1.0, atol=1e-12)
    assert state.placed_count == parameters.N
    assert state.capacity >= parameters.N
    assert state.mirror_vertices.shape == (4 * parameters.N, 3)
    assert state.mirror_faces.shape == (parameters.N, 4)
    assert np.all((state.cosine_efficiency >= 0.0) & (state.cosine_efficiency <= 1.0))
    assert np.all(
        (state.atmospheric_efficiency >= 0.0)
        & (state.atmospheric_efficiency <= 1.0)
    )

    field_radii = np.linalg.norm(state.centres[:, :2], axis=1)
    half_diagonal = 0.5 * math.hypot(
        parameters.mirror_width,
        parameters.mirror_height,
    )
    assert np.max(field_radii) <= parameters.field_radius - half_diagonal - 0.5 + 1e-9

    # The centre ray must obey the law of reflection and point at the receiver.
    incoming = -state.solar.direction[None, :]
    reflected = incoming - 2.0 * np.einsum(
        "ij,ij->i",
        np.broadcast_to(incoming, state.normals.shape),
        state.normals,
    )[:, None] * state.normals
    assert np.allclose(reflected, state.receiver_directions, atol=1e-12)

    constrained = HeliostatParameters(mirror_width=3.0, mirror_height=6.0)
    enforce_constraints(constrained, "mirror_width")
    assert constrained.mirror_height == constrained.mirror_width

    print("自检通过：太阳方向、镜场布局、镜面网格和反射定律均正常。")
    print(format_status(parameters, state))


def launch_gui() -> None:
    if not GUI_AVAILABLE:
        missing = GUI_IMPORT_ERROR.name if GUI_IMPORT_ERROR is not None else "GUI 依赖"
        raise SystemExit(
            f"缺少 Python 包：{missing}\n"
            "请在 Conda agent 环境中运行：\n"
            "conda run -n agent python -m pip install "
            "numpy PySide6 pyvista pyvistaqt"
        )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Heliostat3DApp()
    window.show()
    raise SystemExit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="塔式太阳能定日镜场交互式三维模型"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只检查数值计算，不启动图形界面",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    else:
        launch_gui()


if __name__ == "__main__":
    main()
