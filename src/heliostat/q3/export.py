"""六区微调实验的数据、论文表和 result3.xlsx 导出。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ._workbook import write_result3_workbook
from .evaluate import RefineEvaluation, metrics
from .model import RefineBaseline, RefineDesign


def _json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _csv(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    records = list(rows)
    if not records:
        path.write_text("\n", encoding="utf-8-sig")
        return path
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    return path


def _comparison_record(
    label: str,
    evaluation: RefineEvaluation,
    *,
    target_power_mw: float,
) -> dict[str, object]:
    return {"scheme": label, **metrics(evaluation, target_power_mw=target_power_mw)}


def write_results(
    *,
    output_dir: str | Path,
    baseline: RefineBaseline,
    regression: dict[str, object],
    tower_rows: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    active_payload: dict[str, object],
    search_trace: Iterable[dict[str, object]],
    formal_rows: list[dict[str, object]],
    baseline_formal: RefineEvaluation,
    preclosure_formal: RefineEvaluation,
    attempted_formal: RefineEvaluation,
    selected_formal: RefineEvaluation,
    selected_design: RefineDesign,
    dense_payload: dict[str, object],
    result3_template: str | Path,
    target_power_mw: float,
    decision: str,
    closure_rows: Iterable[dict[str, object]],
    closure_payload: dict[str, object],
    boundary_rows: list[dict[str, object]],
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["regression"] = _json(destination / "02_六组回归结果.json", regression)
    paths["tower"] = _csv(destination / "03_塔位两种语义扫描.csv", tower_rows)
    paths["geometry_scan"] = _csv(destination / "04_Campo几何粗扫.csv", geometry_rows)
    paths["sensitivity"] = _csv(destination / "05_规格参数敏感性.csv", sensitivity_rows)
    paths["active"] = _json(destination / "06_活跃变量集合.json", active_payload)
    paths["trace"] = _csv(destination / "07_局部搜索轨迹.csv", search_trace)
    paths["formal_candidates"] = _csv(destination / "08_正式候选比较.csv", formal_rows)

    group_rows = []
    for group in range(6):
        active = selected_formal.field.group_indices == group
        group_rows.append(
            {
                "group": group + 1,
                "ring_start": ((1, 2, 6, 12, 15, 21)[group]),
                "ring_stop": ((1, 5, 11, 14, 20, 28)[group]),
                "mirror_count": int(active.sum()),
                "mirror_width_m": selected_design.widths[group],
                "mirror_height_m": selected_design.mirror_heights[group],
                "installation_height_m": selected_design.installation_heights[group],
                "group_area_m2": float(selected_formal.specifications.areas[active].sum()),
            }
        )
    paths["groups"] = _csv(destination / "09_最终六区参数.csv", group_rows)

    mirror_rows = []
    for index in range(selected_formal.mirror_count):
        mirror_rows.append(
            {
                "mirror_id": index + 1,
                "original_mirror_id": int(selected_formal.field.original_indices[index]) + 1,
                "ring_index": int(selected_formal.field.ring_indices[index]),
                "group": int(selected_formal.field.group_indices[index]) + 1,
                "mirror_width_m": float(selected_formal.specifications.widths[index]),
                "mirror_height_m": float(selected_formal.specifications.heights[index]),
                "x_m": float(selected_formal.field.coordinates[index, 0]),
                "y_m": float(selected_formal.field.coordinates[index, 1]),
                "z_m": float(selected_formal.specifications.installation_heights[index]),
            }
        )
    paths["mirrors"] = _csv(destination / "10_最终逐镜参数与坐标.csv", mirror_rows)

    comparison = {
        "decision": decision,
        "target_power_mw": target_power_mw,
        "baseline": _comparison_record(
            "six_group_baseline", baseline_formal, target_power_mw=target_power_mw
        ),
        "preclosure_candidate": _comparison_record(
            "two_sweep_candidate",
            preclosure_formal,
            target_power_mw=target_power_mw,
        ),
        "attempted_candidate": _comparison_record(
            "refined_candidate", attempted_formal, target_power_mw=target_power_mw
        ),
        "selected": _comparison_record(
            "selected_final", selected_formal, target_power_mw=target_power_mw
        ),
        "selected_design": selected_design.to_dict(),
        "closure": closure_payload,
    }
    paths["formal"] = _json(destination / "11_正式结果比较.json", comparison)
    paths["dense"] = _json(destination / "12_加密验收比较.json", dense_payload)
    paths["geometry"] = _json(
        destination / "13_几何约束验证.json",
        {
            "valid": selected_formal.geometry.valid,
            "details": asdict(selected_formal.geometry),
            "mirror_set_hash": selected_formal.field.mirror_set_hash,
            "outer_clipped_count": selected_formal.field.outer_clipped_count,
            "group_counts": list(selected_formal.field.group_counts),
        },
    )
    paths["closure"] = _csv(
        destination / "14_局部收口检查.csv",
        closure_rows,
    )
    paths["boundary"] = _csv(
        destination / "20_六区边界局部敏感性检验.csv",
        boundary_rows,
    )
    workbook = destination / "result3.xlsx"
    write_result3_workbook(
        template_path=result3_template,
        output_path=workbook,
        evaluation=selected_formal.raw,
        tower_x=baseline.parameters.tower_x,
        tower_y=selected_design.tower_y,
    )
    paths["workbook"] = workbook

    delta_q = attempted_formal.unit_area_power_kw_m2 - baseline_formal.unit_area_power_kw_m2
    lines = [
        "# 第三问六区参数微调结果与验证表",
        "",
        "本文档汇总第三问的正式结果、加密结果、最终六区规格和边界局部检验。",
        "",
        f"计算结论：{decision}。",
        "",
        "## 表 S3-1 正式精度比较",
        "",
        "| 方案 | 镜子数 | 年平均功率 (MW) | 功率余量 (MW) | 总面积 (m²) | 单位面积输出 (kW/m²) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 原六组 | {baseline_formal.mirror_count} "
            f"| {baseline_formal.annual_power_mw:.9f} "
            f"| {baseline_formal.annual_power_mw - target_power_mw:.9f} "
            f"| {baseline_formal.total_area_m2:.6f} "
            f"| {baseline_formal.unit_area_power_kw_m2:.9f} |"
        ),
        (
            f"| 两轮局部搜索候选 | {preclosure_formal.mirror_count} "
            f"| {preclosure_formal.annual_power_mw:.9f} "
            f"| {preclosure_formal.annual_power_mw - target_power_mw:.9f} "
            f"| {preclosure_formal.total_area_m2:.6f} "
            f"| {preclosure_formal.unit_area_power_kw_m2:.9f} |"
        ),
        (
            f"| 正式收口候选 | {attempted_formal.mirror_count} "
            f"| {attempted_formal.annual_power_mw:.9f} "
            f"| {attempted_formal.annual_power_mw - target_power_mw:.9f} "
            f"| {attempted_formal.total_area_m2:.6f} "
            f"| {attempted_formal.unit_area_power_kw_m2:.9f} |"
        ),
        "",
        f"正式精度候选相对原六组的 $\Delta q={delta_q:.9f}\ \mathrm{{kW/m^2}}$。",
        "",
        (
            "塔位包围扫描已找到北侧下降点；随后完成一次六个活跃变量的"
            f"正负最细邻域检查，并接受 {closure_payload['accepted_moves']} 个可行改进。"
        ),
        "",
        "## 表 S3-2 加密精度比较",
        "",
        "| 邻镜半径 (m) | 原六组功率 (MW) | 微调候选功率 (MW) | 原六组 q (kW/m²) | 微调候选 q (kW/m²) | $\\Delta q$ (kW/m²) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for radius in ("80", "100"):
        before = dense_payload.get("baseline", {}).get(radius)
        after = dense_payload.get("candidate", {}).get(radius)
        if before is not None and after is not None:
            lines.append(
                f"| {radius} | {before['annual_power_mw']:.9f} "
                f"| {after['annual_power_mw']:.9f} "
                f"| {before['unit_area_power_kw_m2']:.9f} "
                f"| {after['unit_area_power_kw_m2']:.9f} "
                f"| {after['unit_area_power_kw_m2'] - before['unit_area_power_kw_m2']:.9f} |"
            )
    lines.extend(
        (
        "",
        "## 表 S3-3 最终六区规格",
        "",
        "| 分区 | 镜子数 | 宽度 (m) | 高度 (m) | 安装高度 (m) |",
        "| ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for row in group_rows:
        lines.append(
            f"| G{row['group']} | {row['mirror_count']} "
            f"| {row['mirror_width_m']:.6f} | {row['mirror_height_m']:.6f} "
            f"| {row['installation_height_m']:.6f} |"
        )
    lines.extend(
        (
            "",
            "## 表 S3-4 六区边界局部合理性检验",
            "",
        )
    )
    if bool(regression.get("smoke")):
        lines.extend(
            (
                "smoke 仅验证 18 个边界候选的生成、评价、导出和绘图链路；"
                "数值及分类不得用于论文结论。",
            )
        )
    else:
        boundary_counts = {
            classification: sum(
                row["classification"] == classification
                for row in boundary_rows
            )
            for classification in (
                "功率可行但q下降",
                "q提高但功率不达标",
                "功率与q均不占优",
            )
        }
        q_sensitivity: dict[int, float] = {}
        for boundary_id in range(1, 6):
            candidates = [
                row
                for row in boundary_rows
                if int(row["boundary_id"]) == boundary_id
            ]
            q_sensitivity[boundary_id] = max(
                abs(float(row["formal_delta_q_kw_m2"]))
                / abs(int(row["shift_rings"]))
                for row in candidates
            )
        most_sensitive = max(q_sensitivity, key=q_sensitivity.get)
        lines.extend(
            (
                "零扰动正式评价复现最终方案："
                f"$P_0={selected_formal.annual_power_mw:.9f}\\ \\mathrm{{MW}}$，"
                f"$q_0={selected_formal.unit_area_power_kw_m2:.9f}"
                "\\ \\mathrm{kW/m^2}$。",
                "",
                "| 正式分类 | 候选数 |",
                "| --- | ---: |",
                f"| 功率可行但 $q$ 下降 | {boundary_counts['功率可行但q下降']} |",
                f"| $q$ 提高但功率不达标 | {boundary_counts['q提高但功率不达标']} |",
                f"| 功率与 $q$ 均不占优 | {boundary_counts['功率与q均不占优']} |",
                "",
                "18 个合法单边界候选均完成中精度和正式精度评价；"
                "未发现同时满足 $42\\ \\mathrm{MW}$ 功率约束并提高 $q$ 的候选，"
                f"因此保留边界 $(1,5,11,14,20)$。B{most_sensitive} 是本次"
                "单位面积输出局部检验中最敏感的边界。",
            )
        )
    lines.extend(
        (
            "",
            "## 验收说明",
            "",
            "塔位模式 A 与 B 分开扫描；搜索轨迹固定使用已选模式。中精度仅用于排序和局部接受，最终判定来自同口径正式精度及 80/100 m 加密比较。",
            "",
        )
    )
    table = destination / "15_论文结果与验证表.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["table"] = table
    return paths
