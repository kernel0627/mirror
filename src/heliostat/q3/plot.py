"""补充方案要求的三张结果图。"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import RefineEvaluation
from .model import RefineBaseline, RefineDesign


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
        (f"{row['parameter']}{row['direction']}", float(row["delta_q"]))
        for row in rows
        if row.get("delta_q") not in (None, "")
    ]
    for row in tower_rows:
        if (
            row.get("tower_mode") == selected_tower_mode
            and abs(float(row.get("delta_y_m", 99.0))) == 0.5
            and row.get("delta_q_from_six_medium") not in (None, "")
        ):
            direction = "+" if float(row["delta_y_m"]) > 0 else "-"
            records.append((f"yT{direction}", float(row["delta_q_from_six_medium"])))
    for row in geometry_rows:
        if row.get("delta_q_from_six_medium") in (None, ""):
            continue
        if row.get("scan") == "D1-one-dimensional" and math.isclose(
            abs(float(row["delta_D1_from_six"])), 0.10, abs_tol=1e-12
        ):
            direction = "+" if float(row["delta_D1_from_six"]) > 0 else "-"
            records.append((f"D1{direction}", float(row["delta_q_from_six_medium"])))
        if row.get("scan") == "g-one-dimensional" and math.isclose(
            abs(float(row["delta_g_from_six"])), 0.01, abs_tol=1e-12
        ):
            direction = "+" if float(row["delta_g_from_six"]) > 0 else "-"
            records.append((f"g{direction}", float(row["delta_q_from_six_medium"])))
    records.sort(key=lambda item: item[1])
    labels = [item[0] for item in records]
    values = [item[1] for item in records]
    figure, axis = plt.subplots(figsize=(9, max(5, 0.22 * len(records))))
    colors = ["#d95f02" if value < 0 else "#1b9e77" for value in values]
    axis.barh(np.arange(len(values)), values, color=colors)
    axis.set_yticks(np.arange(len(values)), labels)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("相对原六组中精度基准的 Δq / (kW/m²)")
    axis.set_title("图 S3-1 塔位、Campo 与六区规格参数敏感性排序")
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


def generate_figures(**kwargs: object) -> tuple[Path, Path, Path]:
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
    )
