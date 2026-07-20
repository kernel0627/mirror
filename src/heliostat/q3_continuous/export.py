"""五节点 Campo 连续模型的精简正式输出。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate import EvaluationProfile, HeterogeneousEvaluation
from .model import CampoMotherField, SplineDesign
from .search import MultiStartOutcome, SearchTraceRecord


TARGET_ANNUAL_POWER_MW = 42.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入 {path.name} 的结果。")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_record(
    evaluation: HeterogeneousEvaluation,
) -> dict[str, Any]:
    return {
        "profile": evaluation.profile_name,
        "mirror_count": evaluation.mirror_count,
        "total_area_m2": evaluation.total_area_m2,
        "annual_power_mw": evaluation.annual_power_mw,
        "annual_power_margin_mw": (
            evaluation.annual_power_mw - TARGET_ANNUAL_POWER_MW
        ),
        "unit_area_power_kw_m2": evaluation.unit_area_power_kw_m2,
        "constraint_satisfied": evaluation.is_feasible(),
        "annual_efficiencies": asdict(evaluation.solution.annual_result),
        "geometry": asdict(evaluation.geometry),
    }


def _trace_row(record: SearchTraceRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sequence": record.sequence,
        "start_name": record.start_name,
        "phase": record.phase,
        "step": record.step,
        "round_index": record.round_index,
        "action": record.action,
        "accepted": record.accepted,
        "feasible": record.feasible,
        "annual_power_mw": record.annual_power_mw,
        "total_area_m2": record.total_area_m2,
        "unit_area_power_kw_m2": record.unit_area_power_kw_m2,
        "area_scale": record.area_scale,
    }
    for index, value in enumerate(record.size_nodes, start=1):
        row[f"alpha_{index}"] = value
    for index, value in enumerate(record.height_nodes, start=1):
        row[f"beta_{index}_m"] = value
    return row


def write_question3_results(
    *,
    output_dir: str | Path,
    mother: CampoMotherField,
    design: SplineDesign,
    evaluation: HeterogeneousEvaluation,
    search: MultiStartOutcome,
    selected_start_name: str,
    formal_evaluations: Sequence[
        tuple[str, HeterogeneousEvaluation]
    ],
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    mirror_path = destination / "02_逐镜最终参数.csv"
    time_path = destination / "03_逐时刻结果.csv"
    monthly_path = destination / "04_月平均结果.csv"
    annual_path = destination / "05_年平均结果.json"
    trace_path = destination / "06_搜索轨迹.csv"
    summary_path = destination / "07_最终方案摘要.json"
    formal_path = destination / "08_正式精度验证.json"
    nodes_path = destination / "10_连续规格节点.csv"

    mirror_rows = [
        {
            "mirror_id": index + 1,
            "original_campo_mirror_id": (
                int(evaluation.original_indices[index]) + 1
            ),
            "ring_index": int(evaluation.ring_indices[index]),
            "radius_to_tower_m": float(evaluation.ring_radii[index]),
            "campo_zone": int(evaluation.zone_indices[index]),
            "nominal_ring_count": int(
                evaluation.nominal_ring_counts[index]
            ),
            "actual_ring_count": int(
                evaluation.actual_ring_counts[index]
            ),
            "x_m": float(evaluation.coordinates[index, 0]),
            "y_m": float(evaluation.coordinates[index, 1]),
            "mirror_width_m": float(evaluation.widths[index]),
            "mirror_height_m": float(evaluation.heights[index]),
            "installation_height_m": float(
                evaluation.installation_heights[index]
            ),
            "mirror_area_m2": float(
                evaluation.widths[index] * evaluation.heights[index]
            ),
            "annual_output_power_kw": float(
                evaluation.solution.mirror_annual_results[
                    index
                ].average_output_power_kw
            ),
        }
        for index in range(evaluation.mirror_count)
    ]
    _write_csv(mirror_path, mirror_rows)
    _write_csv(
        time_path,
        [asdict(record) for record in evaluation.solution.time_results],
    )
    _write_csv(
        monthly_path,
        [asdict(record) for record in evaluation.solution.monthly_results],
    )
    annual_path.write_text(
        json.dumps(
            asdict(evaluation.solution.annual_result),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    trace_rows = [
        _trace_row(record)
        for outcome in search.starts
        for record in outcome.trace
    ]
    _write_csv(trace_path, trace_rows)

    node_rows: list[dict[str, Any]] = []
    for index, (ring, radius, alpha, beta) in enumerate(
        zip(
            mother.control_ring_indices,
            mother.control_radii,
            design.size_nodes,
            design.height_nodes,
        ),
        start=1,
    ):
        active = mother.ring_indices == ring
        actual = int(mother.actual_ring_counts[np.flatnonzero(active)[0]])
        nominal = int(
            mother.nominal_ring_counts[np.flatnonzero(active)[0]]
        )
        node_rows.append(
            {
                "node": index,
                "ring_index": ring,
                "radius_m": radius,
                "actual_ring_count": actual,
                "nominal_ring_count": nominal,
                "retention_ratio": actual / nominal,
                "size_alpha": alpha,
                "installation_height_beta_m": beta,
                "area_scale_lambda": design.area_scale,
            }
        )
    _write_csv(nodes_path, node_rows)

    convergence = [
        {
            "start_name": outcome.start_name,
            "requested_initial_size_nodes": list(
                outcome.requested_initial_design.size_nodes
            ),
            "requested_initial_height_nodes_m": list(
                outcome.requested_initial_design.height_nodes
            ),
            "height_projection_factor": (
                outcome.height_projection_factor
            ),
            "size_projection_factor": outcome.size_projection_factor,
            "joint_cycles": outcome.joint_cycles,
            "stable_joint_cycles": outcome.stable_joint_cycles,
            "stopped_by": outcome.stopped_by,
            "medium_result": _evaluation_record(
                outcome.best_evaluation
            ),
            "final_size_nodes": list(outcome.best_design.size_nodes),
            "final_height_nodes_m": list(
                outcome.best_design.height_nodes
            ),
            "final_area_scale": outcome.best_design.area_scale,
        }
        for outcome in search.starts
    ]
    summary = {
        "layout": "fixed-q2-1469-campo-five-node-radial-spline",
        "tower": {
            "x_m": mother.parameters.tower_x,
            "y_m": mother.parameters.tower_y,
        },
        "control_ring_indices": list(mother.control_ring_indices),
        "control_radii_m": list(mother.control_radii),
        "selected_start": selected_start_name,
        "size_nodes": list(design.size_nodes),
        "height_nodes_m": list(design.height_nodes),
        "area_scale_lambda": design.area_scale,
        "formal_result": _evaluation_record(evaluation),
        "convergence": convergence,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    formal_payload = {
        "selection_rule": (
            "Among candidates satisfying annual_power_mw >= 42, "
            "select maximum unit_area_power_kw_m2."
        ),
        "candidates": [
            {
                "start_name": name,
                **_evaluation_record(candidate),
            }
            for name, candidate in formal_evaluations
        ],
        "selected_start": selected_start_name,
        "selected": _evaluation_record(evaluation),
    }
    formal_path.write_text(
        json.dumps(formal_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "mirrors": mirror_path,
        "time_results": time_path,
        "monthly": monthly_path,
        "annual": annual_path,
        "trace": trace_path,
        "summary": summary_path,
        "formal_validation": formal_path,
        "nodes": nodes_path,
    }


def write_dense_validation(
    *,
    output_dir: str | Path,
    evaluations: Sequence[
        tuple[EvaluationProfile, HeterogeneousEvaluation]
    ],
) -> Path:
    if not evaluations:
        raise ValueError("加密验证结果不能为空。")
    path = Path(output_dir) / "09_加密精度验证.json"
    payload = {
        "evaluations": [
            {
                "profile": {
                    "shadow_grid_size": (
                        profile.solver.shadow_grid_size
                    ),
                    "truncation_rays": (
                        profile.solver.truncation_rays
                    ),
                    "neighbor_radius_m": (
                        profile.solver.neighbor_radius_m
                    ),
                    "state_count": (
                        len(profile.months)
                        * len(profile.solar_times)
                    ),
                },
                **_evaluation_record(evaluation),
            }
            for profile, evaluation in evaluations
        ]
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
