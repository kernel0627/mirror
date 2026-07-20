"""生成第二问两种最终候选布局的四张正式结果图。"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ..config import FieldConfig
from .evaluate import (
    FieldEvaluation,
    LayoutParameters,
    evaluate_coordinates,
    final_profile,
)
from .layout import (
    CampoParameters,
    PartitionedRingParameters,
    generate_partitioned_layout,
)

PARTITIONED_COLOR = "#2563EB"
CAMPO_COLOR = "#D97706"
TARGET_COLOR = "#C2413B"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#172033"
RECEIVER_COLOR = "#C2410C"
RAY_COLOR = "#E76F51"
POWER_CMAP = "viridis"


def configure_matplotlib() -> None:
    font_path = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font_path.exists():
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8A93A3",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": "#465166",
            "ytick.color": "#465166",
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.72,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_partitioned_result(comparison: dict):
    record = comparison["partitioned"]
    parameters = PartitionedRingParameters(**record["parameters"])
    layout = generate_partitioned_layout(parameters)
    coordinates = layout.prefix(int(record["ring_count"]))
    if coordinates.shape[0] != int(record["mirror_count"]):
        raise RuntimeError(
            "分区交错同心圆镜数不一致："
            f"{coordinates.shape[0]} != {record['mirror_count']}"
        )
    evaluation = evaluate_coordinates(
        layout_kind="partitioned",
        ring_count=int(record["ring_count"]),
        coordinates=coordinates,
        parameters=parameters,
        profile=final_profile(),
    )
    if abs(evaluation.annual_power_mw - float(record["annual_power_mw"])) > 1e-9:
        raise RuntimeError("分区交错同心圆正式精度复算未复现 final 结果。")
    return parameters, coordinates, evaluation


def build_campo_result(comparison: dict, output_dir: Path):
    record = comparison["campo"]
    parameters = CampoParameters(**record["parameters"])
    coordinate_rows = load_csv(output_dir / "03_最终镜位坐标.csv")
    coordinates = np.asarray(
        [[float(row["x_m"]), float(row["y_m"])] for row in coordinate_rows],
        dtype=float,
    )
    evaluation = evaluate_coordinates(
        layout_kind="campo",
        ring_count=int(record["ring_count"]),
        coordinates=coordinates,
        parameters=parameters,
        profile=final_profile(),
    )
    if coordinates.shape[0] != int(record["mirror_count"]):
        raise RuntimeError("Campo 坐标镜数与正式摘要不一致。")
    if abs(evaluation.annual_power_mw - float(record["annual_power_mw"])) > 1e-8:
        raise RuntimeError("Campo 正式精度复算未复现交付结果。")
    return parameters, coordinates, evaluation


def add_layout_frame(
    ax,
    *,
    tower_x: float,
    tower_y: float,
    field_radius: float,
    exclusion_radius: float,
) -> None:
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            field_radius,
            fill=False,
            color="#111827",
            linewidth=1.7,
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            (tower_x, tower_y),
            exclusion_radius,
            fill=False,
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=1.4,
            zorder=4,
        )
    )
    ax.scatter(
        [tower_x],
        [tower_y],
        marker="*",
        s=210,
        color=TARGET_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax.set_xlim(-370, 370)
    ax.set_ylim(-370, 370)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / East (m)")
    ax.set_ylabel("y / North (m)")
    ax.grid(True)


def plot_layout_comparison(
    *,
    figure_dir: Path,
    comparison: dict,
    partitioned_parameters: PartitionedRingParameters,
    partitioned_coordinates: np.ndarray,
    partitioned_powers: np.ndarray,
    campo_parameters: CampoParameters,
    campo_coordinates: np.ndarray,
    campo_powers: np.ndarray,
) -> Path:
    all_powers = np.concatenate((partitioned_powers, campo_powers))
    norm = colors.Normalize(
        vmin=float(np.percentile(all_powers, 1.0)),
        vmax=float(np.percentile(all_powers, 99.0)),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.4), constrained_layout=True)
    layouts = (
        (
            axes[0],
            "方案A：分区交错同心圆",
            comparison["partitioned"],
            partitioned_parameters,
            partitioned_coordinates,
            partitioned_powers,
        ),
        (
            axes[1],
            "方案B：改进 Campo",
            comparison["campo"],
            campo_parameters,
            campo_coordinates,
            campo_powers,
        ),
    )
    scatter = None
    for ax, title, record, parameters, coordinates, powers in layouts:
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=powers,
            cmap=POWER_CMAP,
            norm=norm,
            s=8,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        add_layout_frame(
            ax,
            tower_x=parameters.tower_x,
            tower_y=parameters.tower_y,
            field_radius=parameters.field_radius,
            exclusion_radius=parameters.exclusion_radius,
        )
        ax.set_title(
            f"{title}\n"
            f"N = {record['mirror_count']}，"
            f"q = {record['unit_area_power_kw_m2']:.6f} kW/m²",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.text(
            0.02,
            0.025,
            f"年平均功率 {record['annual_power_mw']:.6f} MW\n"
            f"总镜面面积 {record['total_area_m2']:.3f} m²",
            transform=ax.transAxes,
            fontsize=9.3,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#CBD2DF",
                "alpha": 0.92,
            },
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=TARGET_COLOR,
            markeredgecolor="white",
            markersize=13,
            label="吸收塔",
        ),
        Line2D(
            [0],
            [0],
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=1.4,
            label="塔周禁布边界（100 m）",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=1.7,
            label="镜场边界（350 m）",
        ),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8.8)
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8.8)
    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes,
            fraction=0.026,
            pad=0.025,
            shrink=0.88,
        )
        colorbar.set_label("单镜年平均输出热功率 (kW)")
    figure.suptitle(
        "图2-1  两种候选布局的平面分布与单镜年平均输出",
        fontsize=16,
        fontweight="bold",
    )
    path = figure_dir / "11_图2-1_两种候选布局平面分布与单镜年平均输出.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_metric_comparison(
    *,
    figure_dir: Path,
    comparison: dict,
    partitioned_optical: float,
    campo_optical: float,
) -> Path:
    names = ("分区交错同心圆", "改进 Campo")
    palette = (PARTITIONED_COLOR, CAMPO_COLOR)
    metrics = (
        (
            "年平均输出热功率",
            (
                comparison["partitioned"]["annual_power_mw"],
                comparison["campo"]["annual_power_mw"],
            ),
            "MW",
            48.0,
            42.0,
            "{:.3f}",
        ),
        (
            "单位镜面面积年平均输出",
            (
                comparison["partitioned"]["unit_area_power_kw_m2"],
                comparison["campo"]["unit_area_power_kw_m2"],
            ),
            "kW/m²",
            0.75,
            None,
            "{:.4f}",
        ),
        (
            "总镜面面积",
            (
                comparison["partitioned"]["total_area_m2"],
                comparison["campo"]["total_area_m2"],
            ),
            "m²",
            72000.0,
            None,
            "{:,.0f}",
        ),
        (
            "年平均综合光学效率",
            (partitioned_optical, campo_optical),
            "",
            0.78,
            None,
            "{:.4f}",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
    x = np.arange(2)
    for ax, (title, values, unit, upper, benchmark, label_format) in zip(
        axes.flat, metrics, strict=True
    ):
        bars = ax.bar(
            x,
            values,
            width=0.56,
            color=palette,
            edgecolor=("#1D4ED8", "#B45309"),
            linewidth=0.9,
        )
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xticks(x, names)
        ax.set_ylabel(unit)
        ax.set_ylim(0.0, upper)
        ax.grid(axis="y")
        if benchmark is not None:
            ax.axhline(
                benchmark,
                color=TARGET_COLOR,
                linestyle="--",
                linewidth=1.5,
                label="42 MW约束",
            )
            ax.legend(loc="lower right", frameon=False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + upper * 0.025,
                label_format.format(value),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    figure.suptitle(
        "图2-2  两种候选布局的主要性能指标对比",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.94,
        "两种布局均采用60个规定时刻与相同正式计算精度",
        ha="center",
        fontsize=10.5,
        color="#526075",
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.92), h_pad=2.4, w_pad=2.0)
    path = figure_dir / "12_图2-2_两种候选布局主要性能指标对比.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_monthly_comparison(
    *,
    figure_dir: Path,
    partitioned_evaluation,
    campo_evaluation,
    comparison: dict,
) -> Path:
    months = np.arange(1, 13)
    partitioned_monthly = partitioned_evaluation.solution.monthly_results
    partitioned_power = np.asarray([row.field_output_mw for row in partitioned_monthly])
    campo_monthly = campo_evaluation.solution.monthly_results
    campo_power = np.asarray([row.field_output_mw for row in campo_monthly])
    partitioned_unit = np.asarray(
        [row.unit_area_output_kw_m2 for row in partitioned_monthly]
    )
    campo_unit = np.asarray([row.unit_area_output_kw_m2 for row in campo_monthly])
    partitioned_optical = np.asarray(
        [row.average_optical_efficiency for row in partitioned_monthly]
    )
    campo_optical = np.asarray(
        [row.average_optical_efficiency for row in campo_monthly]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12.6, 10.6),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0, 1.0)},
    )
    width = 0.36
    axes[0].bar(
        months - width / 2,
        partitioned_power,
        width,
        color=PARTITIONED_COLOR,
        edgecolor="#1D4ED8",
        linewidth=0.7,
        label=(
            "分区交错同心圆"
            f"（年均 {comparison['partitioned']['annual_power_mw']:.3f} MW）"
        ),
    )
    axes[0].bar(
        months + width / 2,
        campo_power,
        width,
        color=CAMPO_COLOR,
        edgecolor="#B45309",
        linewidth=0.7,
        label=(f"改进 Campo（年均 {comparison['campo']['annual_power_mw']:.3f} MW）"),
    )
    axes[0].axhline(
        42.0,
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="42 MW约束",
    )
    axes[0].set_ylabel("热功率 (MW)")
    axes[0].set_title("月平均输出热功率", fontsize=12.5, fontweight="bold")
    axes[0].set_ylim(0, max(partitioned_power.max(), campo_power.max()) * 1.18)
    axes[0].legend(ncol=3, loc="upper center", fontsize=9)
    axes[0].grid(axis="y")

    axes[1].plot(
        months,
        partitioned_unit,
        color=PARTITIONED_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="分区交错同心圆",
    )
    axes[1].plot(
        months,
        campo_unit,
        color=CAMPO_COLOR,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=5,
        label="改进 Campo",
    )
    axes[1].set_ylabel("单位面积功率\n(kW/m²)")
    axes[1].set_title(
        "单位镜面面积月平均输出热功率",
        fontsize=12.5,
        fontweight="bold",
    )
    axes[1].grid(True)
    axes[1].legend(loc="lower center", ncol=2)

    axes[2].plot(
        months,
        partitioned_optical,
        color=PARTITIONED_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="分区交错同心圆",
    )
    axes[2].plot(
        months,
        campo_optical,
        color=CAMPO_COLOR,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=5,
        label="改进 Campo",
    )
    axes[2].set_xlabel("月份")
    axes[2].set_ylabel("综合光学效率")
    axes[2].set_title("月平均综合光学效率", fontsize=12.5, fontweight="bold")
    axes[2].set_xticks(months)
    optical_min = min(partitioned_optical.min(), campo_optical.min())
    optical_max = max(partitioned_optical.max(), campo_optical.max())
    axes[2].set_ylim(
        max(0.0, optical_min - 0.02),
        min(1.0, optical_max + 0.02),
    )
    axes[2].grid(True)
    axes[2].legend(loc="lower center", ncol=2)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    figure.suptitle(
        "图2-3  两种候选布局的月平均性能对比",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.96), h_pad=1.6)
    path = figure_dir / "13_图2-3_两种候选布局月平均性能对比.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def representative_indices(
    coordinates: np.ndarray,
    tower_x: float,
    tower_y: float,
    count: int = 20,
) -> np.ndarray:
    angles = np.arctan2(
        coordinates[:, 1] - tower_y,
        coordinates[:, 0] - tower_x,
    )
    order = np.argsort(angles)
    picks = np.linspace(0, len(order) - 1, count, dtype=int)
    return order[picks]


def draw_receiver(ax, config: FieldConfig) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    z_values = np.linspace(config.receiver_z_min, config.receiver_z_max, 8)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x_grid = config.tower_x + config.receiver_radius * np.cos(theta_grid)
    y_grid = config.tower_y + config.receiver_radius * np.sin(theta_grid)
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=RECEIVER_COLOR,
        alpha=0.9,
        linewidth=0,
        shade=True,
    )
    ax.plot(
        [config.tower_x, config.tower_x],
        [config.tower_y, config.tower_y],
        [0.0, config.receiver_z_min],
        color="#667085",
        linewidth=4.5,
        solid_capstyle="round",
    )


def plot_3d_comparison(
    *,
    figure_dir: Path,
    partitioned_parameters: PartitionedRingParameters,
    partitioned_coordinates: np.ndarray,
    partitioned_powers: np.ndarray,
    campo_parameters: CampoParameters,
    campo_coordinates: np.ndarray,
    campo_powers: np.ndarray,
) -> Path:
    all_powers = np.concatenate((partitioned_powers, campo_powers))
    norm = colors.Normalize(
        vmin=float(np.percentile(all_powers, 1.0)),
        vmax=float(np.percentile(all_powers, 99.0)),
    )
    figure = plt.figure(figsize=(16.0, 8.0))
    axes = (
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    )
    layouts = (
        (
            axes[0],
            "方案A：分区交错同心圆",
            partitioned_parameters,
            partitioned_coordinates,
            partitioned_powers,
        ),
        (
            axes[1],
            "方案B：改进 Campo",
            campo_parameters,
            campo_coordinates,
            campo_powers,
        ),
    )
    scatter = None
    for ax, title, parameters, coordinates, powers in layouts:
        config = replace(
            FieldConfig(),
            tower_x=parameters.tower_x,
            tower_y=parameters.tower_y,
            mirror_width=parameters.mirror_width,
            mirror_height=parameters.mirror_height,
            mirror_center_z=parameters.installation_height,
        )
        z = np.full(coordinates.shape[0], config.mirror_center_z)
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            z,
            c=powers,
            cmap=POWER_CMAP,
            norm=norm,
            s=7,
            linewidths=0,
            alpha=0.95,
            rasterized=True,
        )
        draw_receiver(ax, config)
        selected = representative_indices(
            coordinates,
            config.tower_x,
            config.tower_y,
        )
        for index in selected:
            ax.plot(
                [coordinates[index, 0], config.tower_x],
                [coordinates[index, 1], config.tower_y],
                [config.mirror_center_z, config.receiver_center_z],
                color=RAY_COLOR,
                alpha=0.32,
                linewidth=0.7,
            )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("x / East (m)", labelpad=8)
        ax.set_ylabel("y / North (m)", labelpad=8)
        ax.set_zlabel("z (m)", labelpad=6)
        ax.set_xlim(-360, 360)
        ax.set_ylim(-360, 360)
        ax.set_zlim(0, 105)
        ax.view_init(elev=27, azim=-61)
        ax.set_box_aspect((1.0, 1.0, 0.33))
        ax.grid(True)
    if scatter is not None:
        colorbar_axis = figure.add_axes((0.925, 0.20, 0.014, 0.60))
        colorbar = figure.colorbar(
            scatter,
            cax=colorbar_axis,
        )
        colorbar.set_label("单镜年平均输出热功率 (kW)")
    figure.suptitle(
        "图2-4  两种候选布局的三维镜场与代表性中心光路",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        "光路线仅用于展示镜位至吸收器中心的空间关系",
        ha="center",
        fontsize=10,
        color="#526075",
    )
    figure.subplots_adjust(left=0.02, right=0.87, bottom=0.04, top=0.89, wspace=0.08)
    path = figure_dir / "14_图2-4_两种候选布局三维镜场与代表性中心光路.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def build_question2_figures(
    *,
    output_dir: str | Path,
    comparison: dict,
    parameters: dict[str, LayoutParameters],
    evaluations: dict[str, FieldEvaluation],
) -> tuple[Path, ...]:
    """由两种候选布局的正式复算对象生成四张论文图。"""

    configure_matplotlib()
    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    partitioned_parameters = parameters["partitioned"]
    campo_parameters = parameters["campo"]
    partitioned_evaluation = evaluations["partitioned"]
    campo_evaluation = evaluations["campo"]
    partitioned_coordinates = partitioned_evaluation.coordinates
    campo_coordinates = campo_evaluation.coordinates
    partitioned_powers = np.asarray(
        [
            row.average_output_power_kw
            for row in partitioned_evaluation.solution.mirror_annual_results
        ],
        dtype=float,
    )
    campo_powers = np.asarray(
        [
            row.average_output_power_kw
            for row in campo_evaluation.solution.mirror_annual_results
        ],
        dtype=float,
    )

    return (
        plot_layout_comparison(
            figure_dir=figure_dir,
            comparison=comparison,
            partitioned_parameters=partitioned_parameters,
            partitioned_coordinates=partitioned_coordinates,
            partitioned_powers=partitioned_powers,
            campo_parameters=campo_parameters,
            campo_coordinates=campo_coordinates,
            campo_powers=campo_powers,
        ),
        plot_metric_comparison(
            figure_dir=figure_dir,
            comparison=comparison,
            partitioned_optical=(
                partitioned_evaluation.solution.annual_result.average_optical_efficiency
            ),
            campo_optical=(
                campo_evaluation.solution.annual_result.average_optical_efficiency
            ),
        ),
        plot_monthly_comparison(
            figure_dir=figure_dir,
            partitioned_evaluation=partitioned_evaluation,
            campo_evaluation=campo_evaluation,
            comparison=comparison,
        ),
        plot_3d_comparison(
            figure_dir=figure_dir,
            partitioned_parameters=partitioned_parameters,
            partitioned_coordinates=partitioned_coordinates,
            partitioned_powers=partitioned_powers,
            campo_parameters=campo_parameters,
            campo_coordinates=campo_coordinates,
            campo_powers=campo_powers,
        ),
    )


def build_question2_figures_from_output(
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """读取扁平交付目录，统一正式精度复算后重新生成四张图。"""

    destination = Path(output_dir)
    comparison = load_json(destination / "02_双布局比较.json")
    (
        partitioned_parameters,
        _,
        partitioned_evaluation,
    ) = build_partitioned_result(comparison)
    campo_parameters, _, campo_evaluation = build_campo_result(
        comparison,
        destination,
    )
    return build_question2_figures(
        output_dir=destination,
        comparison=comparison,
        parameters={
            "partitioned": partitioned_parameters,
            "campo": campo_parameters,
        },
        evaluations={
            "partitioned": partitioned_evaluation,
            "campo": campo_evaluation,
        },
    )
