"""第三问敏感性、规格、指标和最终镜场结果图。"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch

from .evaluate import RefineEvaluation
from .model import RefineBaseline, RefineDesign


def configure_matplotlib() -> None:
    font_candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for font_path in font_candidates:
        if not font_path.exists():
            continue
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(
            fname=font_path
        ).get_name()
        plt.rcParams["font.family"] = font_name
        break
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "PingFang SC",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "WenQuanYi Zen Hei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_sensitivity(
    rows: list[dict[str, object]],
    *,
    tower_rows: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
    selected_tower_mode: str,
    output_dir: str | Path,
) -> Path:
    records = [
        (
            "六区规格",
            f"{row['parameter']}{row['direction']}",
            float(row["delta_q_from_geometry"]),
        )
        for row in rows
        if row.get("delta_q_from_geometry") not in (None, "")
    ]
    for row in tower_rows:
        if (
            row.get("tower_mode") == selected_tower_mode
            and abs(float(row.get("delta_y_m", 99.0))) == 0.5
            and row.get("delta_q_from_six_medium") not in (None, "")
        ):
            direction = "+" if float(row["delta_y_m"]) > 0 else "-"
            records.append(
                (
                    "塔位",
                    f"yT{direction}",
                    float(row["delta_q_from_six_medium"]),
                )
            )
    for row in geometry_rows:
        if row.get("delta_q_from_tower_medium") in (None, ""):
            continue
        if row.get("scan") == "D1-one-dimensional" and math.isclose(
            abs(float(row["delta_D1_from_six"])), 0.10, abs_tol=1e-12
        ):
            direction = "+" if float(row["delta_D1_from_six"]) > 0 else "-"
            records.append(
                (
                    "Campo",
                    f"D1{direction}",
                    float(row["delta_q_from_tower_medium"]),
                )
            )
        if row.get("scan") == "g-one-dimensional" and math.isclose(
            abs(float(row["delta_g_from_six"])), 0.01, abs_tol=1e-12
        ):
            direction = "+" if float(row["delta_g_from_six"]) > 0 else "-"
            records.append(
                (
                    "Campo",
                    f"g{direction}",
                    float(row["delta_q_from_tower_medium"]),
                )
            )
    stage_order = {"塔位": 0, "Campo": 1, "六区规格": 2}
    records.sort(key=lambda item: (stage_order[item[0]], item[2]))
    labels = [f"{item[0]} · {item[1]}" for item in records]
    values = [item[2] for item in records]
    stage_colors = {
        "塔位": "#7570b3",
        "Campo": "#d95f02",
        "六区规格": "#1b9e77",
    }
    figure, axis = plt.subplots(figsize=(9, max(5, 0.22 * len(records))))
    colors = [stage_colors[item[0]] for item in records]
    axis.barh(np.arange(len(values)), values, color=colors)
    axis.set_yticks(np.arange(len(values)), labels)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("相对于对应阶段基准的 Δq / (kW/m²)")
    axis.set_title("图 S3-1 各阶段候选相对于对应阶段基准的单位面积输出变化")
    axis.legend(
        handles=[
            Patch(color=color, label=stage)
            for stage, color in stage_colors.items()
        ],
        loc="best",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = Path(output_dir) / "16_参数敏感性图.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_group_parameters(
    baseline: RefineBaseline,
    selected: RefineDesign,
    output_dir: str | Path,
) -> Path:
    groups = np.arange(1, 7)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    series = (
        (baseline.design.widths, selected.widths, "镜宽 / m"),
        (baseline.design.mirror_heights, selected.mirror_heights, "镜高 / m"),
        (
            baseline.design.installation_heights,
            selected.installation_heights,
            "安装高度 / m",
        ),
    )
    for axis, (before, after, ylabel) in zip(axes, series):
        axis.plot(groups, before, "o--", label="原六组")
        axis.plot(groups, after, "s-", label="微调后")
        axis.set_xlabel("径向分区")
        axis.set_ylabel(ylabel)
        axis.set_xticks(groups)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("图 S3-2 优化前后六区规格对比")
    figure.tight_layout()
    path = Path(output_dir) / "17_六区宽高与安装高度图.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_metric_comparison(
    *,
    baseline_formal: RefineEvaluation,
    candidate_formal: RefineEvaluation,
    dense_payload: dict[str, object],
    output_dir: str | Path,
) -> Path:
    labels = ["正式 q", "80 m q", "100 m q"]
    baseline_dense = dense_payload.get("baseline", {})
    candidate_dense = dense_payload.get("candidate", {})
    before = [
        baseline_formal.unit_area_power_kw_m2,
        float(baseline_dense.get("80", {}).get("unit_area_power_kw_m2", np.nan)),
        float(baseline_dense.get("100", {}).get("unit_area_power_kw_m2", np.nan)),
    ]
    after = [
        candidate_formal.unit_area_power_kw_m2,
        float(candidate_dense.get("80", {}).get("unit_area_power_kw_m2", np.nan)),
        float(candidate_dense.get("100", {}).get("unit_area_power_kw_m2", np.nan)),
    ]
    x = np.arange(3)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x - width / 2, before, width, label="原六组")
    axes[0].bar(x + width / 2, after, width, label="微调候选")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("q / (kW/m²)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    powers = [baseline_formal.annual_power_mw, candidate_formal.annual_power_mw]
    axes[1].bar(("原六组", "微调候选"), powers, color=("#7570b3", "#1b9e77"))
    axes[1].axhline(42.0, color="#d95f02", linestyle="--", label="42 MW 约束")
    axes[1].set_ylabel("正式年平均功率 / MW")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("图 S3-3 正式与加密结果比较")
    figure.tight_layout()
    path = Path(output_dir) / "18_六组与优化方案指标比较图.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_final_field(
    *,
    baseline: RefineBaseline,
    selected: RefineDesign,
    evaluation: RefineEvaluation,
    output_dir: str | Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.2, 8.2))
    colors = ("#2166AC", "#67A9CF", "#D1E5F0", "#FDDBC7", "#EF8A62", "#B2182B")
    for group, color in enumerate(colors):
        active = evaluation.field.group_indices == group
        axis.scatter(
            evaluation.field.coordinates[active, 0],
            evaluation.field.coordinates[active, 1],
            s=5,
            color=color,
            alpha=0.78,
            label=f"G{group + 1}",
            rasterized=True,
        )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            baseline.parameters.field_radius,
            fill=False,
            color="#475569",
            linewidth=1.2,
            label="350 m 场地边界",
        )
    )
    axis.add_patch(
        Circle(
            (baseline.parameters.tower_x, selected.tower_y),
            baseline.parameters.exclusion_radius,
            fill=False,
            color="#F59E0B",
            linestyle="--",
            linewidth=1.2,
            label="最终塔周 100 m 禁区",
        )
    )
    axis.scatter(
        (baseline.parameters.tower_x,),
        (baseline.design.tower_y,),
        marker="x",
        s=90,
        linewidths=2.0,
        color="#111827",
        label="原塔位",
        zorder=5,
    )
    axis.scatter(
        (baseline.parameters.tower_x,),
        (selected.tower_y,),
        marker="*",
        s=145,
        color="#DC2626",
        edgecolors="white",
        linewidths=0.7,
        label="最终塔位",
        zorder=6,
    )
    displacement = selected.tower_y - baseline.design.tower_y
    axis.annotate(
        f"向北移动 {displacement:.1f} m",
        xy=(baseline.parameters.tower_x, selected.tower_y),
        xytext=(30.0, baseline.design.tower_y - 18.0),
        arrowprops={"arrowstyle": "->", "color": "#DC2626"},
        color="#991B1B",
        fontsize=10,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-365.0, 365.0)
    axis.set_ylim(-365.0, 365.0)
    axis.set_xlabel("东西坐标 x / m")
    axis.set_ylabel("南北坐标 y / m")
    axis.set_title("图 3-1 最终六区镜场、场地边界与塔位变化")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    figure.tight_layout()
    path = Path(output_dir) / "19_最终六区镜场与塔位平面图.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path


def plot_boundary_sensitivity(
    rows: list[dict[str, object]],
    *,
    output_dir: str | Path,
) -> Path:
    """绘制全部单边界候选的正式 Δq、功率余量和分类。"""

    configure_matplotlib()
    labels = [str(row["candidate"]) for row in rows]
    positions = np.arange(len(labels))
    category_order = (
        "功率可行但q下降",
        "q提高但功率不达标",
        "功率与q均不占优",
        "smoke仅验证链路",
    )
    colors = {
        "功率可行但q下降": "#2A9D8F",
        "q提高但功率不达标": "#E76F51",
        "功率与q均不占优": "#7A7A7A",
        "smoke仅验证链路": "#5E60CE",
    }
    categories = tuple(
        category
        for category in category_order
        if any(row["classification"] == category for row in rows)
    )
    bar_colors = [colors[str(row["classification"])] for row in rows]
    delta_q = [float(row["formal_delta_q_kw_m2"]) for row in rows]
    power_margin = [float(row["formal_power_margin_mw"]) for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    axes[0].bar(positions, delta_q, color=bar_colors, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("正式 Δq / (kW/m²)")
    axes[0].set_title("图 S3-5 六区边界单因素局部敏感性检验")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(
        handles=[
            Patch(color=colors[category], label=category)
            for category in categories
        ],
        loc="best",
    )

    axes[1].bar(positions, power_margin, color=bar_colors, alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("正式功率余量 / MW")
    axes[1].set_xlabel("边界候选（B1--B5，数字为移动环数）")
    axes[1].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    path = Path(output_dir) / "21_六区边界局部敏感性图.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def generate_figures(**kwargs: object) -> tuple[Path, Path, Path, Path]:
    configure_matplotlib()
    return (
        plot_sensitivity(
            kwargs["sensitivity_rows"],
            tower_rows=kwargs["tower_rows"],
            geometry_rows=kwargs["geometry_rows"],
            selected_tower_mode=kwargs["selected_tower_mode"],
            output_dir=kwargs["output_dir"],
        ),
        plot_group_parameters(kwargs["baseline"], kwargs["selected_design"], kwargs["output_dir"]),
        plot_metric_comparison(
            baseline_formal=kwargs["baseline_formal"],
            candidate_formal=kwargs["candidate_formal"],
            dense_payload=kwargs["dense_payload"],
            output_dir=kwargs["output_dir"],
        ),
        plot_final_field(
            baseline=kwargs["baseline"],
            selected=kwargs["selected_design"],
            evaluation=kwargs["selected_formal"],
            output_dir=kwargs["output_dir"],
        ),
    )
