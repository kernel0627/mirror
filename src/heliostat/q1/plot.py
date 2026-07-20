"""第一问的两张正式结果图。"""

# ruff: noqa: E402

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

_mpl_config = Path(tempfile.gettempdir()) / "cowork-matplotlib"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


BLUE = "#2F5D7C"
BLUE_DARK = "#173B54"
BLUE_LIGHT = "#DCE9F1"
ORANGE = "#D97706"
DARK = "#24323D"
GREY = "#76838F"
LIGHT_GREY = "#D7DEE3"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Hiragino Sans GB",
                "Arial Unicode MS",
                "PingFang SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_monthly_performance(output_dir: str | Path) -> Path:
    """绘制月平均综合光学效率和单位面积输出热功率。"""

    _configure_style()
    destination = Path(output_dir)
    rows = _read_csv(destination / "03_月平均计算结果.csv")
    months = np.array([int(row["month"]) for row in rows])
    optical = np.array(
        [float(row["average_optical_efficiency"]) for row in rows]
    )
    unit_power = np.array(
        [float(row["unit_area_output_kw_m2"]) for row in rows]
    )
    output_path = destination / "08_月平均光学性能与输出热功率.png"

    fig, (ax_efficiency, ax_power) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.3),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.15), "hspace": 0.12},
    )

    ax_efficiency.plot(
        months,
        optical,
        color=BLUE,
        linewidth=2.2,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    efficiency_padding = 0.02
    ax_efficiency.set_ylim(
        max(0.0, float(np.min(optical)) - efficiency_padding),
        min(1.0, float(np.max(optical)) + efficiency_padding),
    )
    ax_efficiency.set_ylabel("综合光学效率")
    ax_efficiency.grid(axis="y")
    ax_efficiency.spines[["top", "right"]].set_visible(False)

    ax_power.bar(
        months,
        unit_power,
        width=0.62,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_power.set_ylim(0.0, float(np.max(unit_power)) * 1.14)
    ax_power.set_ylabel(
        r"单位面积输出热功率 ($\mathrm{kW\,m^{-2}}$)"
    )
    ax_power.set_xlabel("月份")
    ax_power.set_xticks(months)
    ax_power.grid(axis="y")
    ax_power.set_axisbelow(True)
    ax_power.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "月平均光学性能与输出热功率",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mirror_annual_efficiency_map(output_dir: str | Path) -> Path:
    """绘制 1745 面定日镜的年平均综合光学效率空间分布。"""

    _configure_style()
    destination = Path(output_dir)
    rows = _read_csv(destination / "05_单镜年平均结果.csv")
    x = np.array([float(row["x_m"]) for row in rows])
    y = np.array([float(row["y_m"]) for row in rows])
    optical = np.array(
        [float(row["average_optical_efficiency"]) for row in rows]
    )
    output_path = destination / "09_单镜年平均综合光学效率空间分布.png"

    efficiency_cmap = LinearSegmentedColormap.from_list(
        "heliostat_efficiency",
        (BLUE_LIGHT, "#8EB7CF", BLUE, BLUE_DARK),
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    points = ax.scatter(
        x,
        y,
        c=optical,
        cmap=efficiency_cmap,
        vmin=float(np.min(optical)),
        vmax=float(np.max(optical)),
        s=18,
        linewidths=0,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=190,
        color=ORANGE,
        edgecolor=DARK,
        linewidth=0.8,
        label="吸收塔",
        zorder=4,
    )
    limit = max(float(np.max(np.abs(x))), float(np.max(np.abs(y)))) + 20.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x 坐标 (m)")
    ax.set_ylabel("y 坐标 (m)")
    ax.set_title("单镜年平均综合光学效率空间分布", fontsize=15, pad=12)
    ax.grid(color=LIGHT_GREY, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    colorbar = fig.colorbar(points, ax=ax, pad=0.025, fraction=0.047)
    colorbar.set_label("年平均综合光学效率")
    colorbar.outline.set_edgecolor(GREY)
    colorbar.outline.set_linewidth(0.7)

    fig.subplots_adjust(left=0.11, right=0.91, top=0.91, bottom=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_paper_figures(output_dir: str | Path) -> dict[str, Path]:
    """生成第一问最终采用的两张结果图。"""

    return {
        "monthly_performance": plot_monthly_performance(output_dir),
        "mirror_efficiency_map": plot_mirror_annual_efficiency_map(
            output_dir
        ),
    }
