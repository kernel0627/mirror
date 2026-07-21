"""第三问 Campo2D 四张正式论文图片。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .evaluate import Campo2DEvaluation


def _is_smoke(evaluation: Campo2DEvaluation) -> bool:
    return "smoke" in evaluation.profile_name.lower()


def _figure_prefix(evaluation: Campo2DEvaluation) -> str:
    return "SMOKE 非正式验证｜" if _is_smoke(evaluation) else ""


def configure_matplotlib() -> None:
    font_path = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font_path.exists():
        matplotlib.font_manager.fontManager.addfont(font_path)
        name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _frame(ax, evaluation: Campo2DEvaluation) -> None:
    parameters = evaluation.field.parameters
    ax.add_patch(
        Circle((0.0, 0.0), parameters.field_radius, fill=False, color="#111827")
    )
    ax.add_patch(
        Circle(
            (parameters.tower_x, parameters.tower_y),
            parameters.exclusion_radius,
            fill=False,
            linestyle="--",
            color="#B91C1C",
        )
    )
    ax.scatter(
        [parameters.tower_x],
        [parameters.tower_y],
        marker="*",
        s=150,
        color="#B91C1C",
        zorder=4,
    )
    ax.set_xlim(-365, 365)
    ax.set_ylim(-365, 365)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.25)


def plot_spatial_specs(
    evaluation: Campo2DEvaluation,
    output_dir: str | Path,
) -> Path:
    coordinates = evaluation.field.coordinates
    specs = evaluation.specifications
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, values, title, label in (
        (axes[0], specs.areas, "单镜面积空间分布", "镜面面积 / m²"),
        (
            axes[1],
            specs.installation_heights,
            "安装高度空间分布",
            "安装高度 / m",
        ),
    ):
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            cmap="viridis",
            s=9,
            linewidths=0,
            rasterized=True,
        )
        _frame(ax, evaluation)
        ax.set_title(title, fontweight="bold")
        figure.colorbar(scatter, ax=ax, shrink=0.82, label=label)
    figure.suptitle(
        _figure_prefix(evaluation) + "图3-1 最终镜面面积与安装高度空间分布",
        fontweight="bold",
    )
    path = Path(output_dir) / "17_图3-1_镜面面积与安装高度空间分布.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_radial_angular(
    evaluation: Campo2DEvaluation,
    output_dir: str | Path,
) -> Path:
    field = evaluation.field
    specs = evaluation.specifications
    rings = np.arange(1, field.ring_count + 1)
    mean_area = np.asarray(
        [np.mean(specs.areas[field.ring_indices == ring]) for ring in rings]
    )
    mean_height = np.asarray(
        [
            np.mean(specs.installation_heights[field.ring_indices == ring])
            for ring in rings
        ]
    )
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9), constrained_layout=True)
    left = axes[0]
    right = left.twinx()
    left.plot(rings, mean_area, "o-", color="#2563EB", label="平均镜面面积")
    right.plot(rings, mean_height, "s-", color="#D97706", label="平均安装高度")
    left.set_xlabel("圆环编号")
    left.set_ylabel("平均镜面面积 / m²", color="#2563EB")
    right.set_ylabel("平均安装高度 / m", color="#D97706")
    left.grid(alpha=0.3)
    left.set_title("径向平均趋势", fontweight="bold")

    representative = (
        field.control_ring_indices[0],
        field.control_ring_indices[2],
        field.control_ring_indices[-1],
    )
    for ring in representative:
        active = field.ring_indices == ring
        order = np.argsort(field.azimuth_angles[active])
        theta = field.azimuth_angles[active][order]
        axes[1].plot(
            theta,
            specs.scales[active][order],
            marker=".",
            linewidth=1.2,
            label=f"第 {ring} 环尺度",
        )
    axes[1].set_xlabel(r"角度 $\theta$ / rad")
    axes[1].set_ylabel("镜面尺度")
    axes[1].set_title("代表性圆环的角度变化", fontweight="bold")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3, fontsize=9)
    figure.suptitle(
        _figure_prefix(evaluation) + "图3-2 径向趋势与代表性圆环角度变化",
        fontweight="bold",
    )
    path = Path(output_dir) / "18_图3-2_径向趋势与代表圆环角度变化.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_q2_q3_mirror_power(
    evaluation: Campo2DEvaluation,
    *,
    q2_mirror_path: str | Path,
    output_dir: str | Path,
) -> Path:
    q2 = _read_csv(q2_mirror_path)
    q2_xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in q2])
    q2_power = np.asarray([float(row["average_output_power_kw"]) for row in q2])
    q3_power = np.asarray(
        [row.average_output_power_kw for row in evaluation.solution.mirror_annual_results]
    )
    values = np.concatenate((q2_power, q3_power))
    lower, upper = np.percentile(values, (1.0, 99.0))
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    scatter = axes[0].scatter(
        q2_xy[:, 0],
        q2_xy[:, 1],
        c=q2_power,
        vmin=lower,
        vmax=upper,
        cmap="plasma",
        s=9,
        linewidths=0,
    )
    axes[0].set_title("问题二统一规格", fontweight="bold")
    axes[0].set_aspect("equal")
    axes[1].scatter(
        evaluation.field.coordinates[:, 0],
        evaluation.field.coordinates[:, 1],
        c=q3_power,
        vmin=lower,
        vmax=upper,
        cmap="plasma",
        s=9,
        linewidths=0,
    )
    q3_title = "第三问径向—角度连续规格"
    if _is_smoke(evaluation):
        q3_title += "（单时刻 smoke）"
    axes[1].set_title(q3_title, fontweight="bold")
    axes[1].set_aspect("equal")
    for ax in axes:
        ax.set_xlim(-365, 365)
        ax.set_ylim(-365, 365)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axes, shrink=0.82, label="单镜年平均输出 / kW")
    figure.suptitle(
        _figure_prefix(evaluation) + "图3-3 问题二与第三问单镜平均输出空间比较",
        fontweight="bold",
    )
    path = Path(output_dir) / "19_图3-3_问题二与第三问单镜年平均输出空间比较.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_baseline_monthly(
    evaluation: Campo2DEvaluation,
    *,
    q2_monthly_path: str | Path,
    baseline_comparison_path: str | Path,
    output_dir: str | Path,
) -> Path:
    comparison = json.loads(Path(baseline_comparison_path).read_text(encoding="utf-8"))
    q2 = comparison["q2_uniform"]
    six = comparison["six_group"]
    new = comparison["q3_campo2d"]
    new_label = "新 Campo2D（smoke）" if _is_smoke(evaluation) else "新 Campo2D"
    labels = ("问题二", "六组 baseline", new_label)

    def metric(record: dict, name: str) -> float:
        if name in record:
            return float(record[name])
        if name == "annual_power_mw":
            return float(record["annual"]["field_output_mw"])
        if name == "unit_area_power_kw_m2":
            return float(record["annual"]["unit_area_output_kw_m2"])
        return float(record[name])

    q2_monthly = _read_csv(q2_monthly_path)
    months = np.asarray([int(row["month"]) for row in q2_monthly])
    q2_unit = np.asarray([float(row["unit_area_output_kw_m2"]) for row in q2_monthly])
    q3_months = np.asarray(
        [row.month for row in evaluation.solution.monthly_results]
    )
    q3_unit = np.asarray(
        [row.unit_area_output_kw_m2 for row in evaluation.solution.monthly_results]
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    x = np.arange(3)
    q_values = [metric(record, "unit_area_power_kw_m2") for record in (q2, six, new)]
    bars = axes[0].bar(x, q_values, color=("#6B7280", "#D97706", "#2563EB"))
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("单位面积输出 / kW/m²")
    axes[0].set_title(
        "baseline 与 smoke 指标（不可作正式比较）"
        if _is_smoke(evaluation)
        else "正式指标比较",
        fontweight="bold",
    )
    axes[0].grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, q_values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.6f}", ha="center", va="bottom")
    axes[1].plot(months, q2_unit, "o-", label="问题二统一规格")
    axes[1].plot(q3_months, q3_unit, "s-", label=new_label)
    axes[1].set_xticks(np.arange(1, 13))
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("月平均单位面积输出 / kW/m²")
    axes[1].set_title("全年月平均表现", fontweight="bold")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.suptitle(
        _figure_prefix(evaluation) + "图3-4 baseline 与月平均性能比较",
        fontweight="bold",
    )
    path = Path(output_dir) / "20_图3-4_baseline与月平均性能比较.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def generate_figures(
    evaluation: Campo2DEvaluation,
    *,
    q2_mirror_path: str | Path,
    q2_monthly_path: str | Path,
    baseline_comparison_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    configure_matplotlib()
    return (
        plot_spatial_specs(evaluation, output_dir),
        plot_radial_angular(evaluation, output_dir),
        plot_q2_q3_mirror_power(
            evaluation,
            q2_mirror_path=q2_mirror_path,
            output_dir=output_dir,
        ),
        plot_baseline_monthly(
            evaluation,
            q2_monthly_path=q2_monthly_path,
            baseline_comparison_path=baseline_comparison_path,
            output_dir=output_dir,
        ),
    )
