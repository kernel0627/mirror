"""从第三问六组正式解出发，执行十组中精度局部细化审计。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from heliostat.q3.evaluate import (
    EvaluationCache,
    EvaluationProfile,
    HeterogeneousEvaluation,
    dense_profile,
    evaluate_specifications,
    field_config_from_mother,
    formal_profile,
    medium_profile,
)
from heliostat.q3.model import (
    CampoMotherField,
    ExpandedSpecifications,
    GroupDesign,
    build_campo_mother_field,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Q2_SUMMARY = (
    PROJECT_ROOT / "outputs" / "q2" / "07_最终方案摘要.json"
)
DEFAULT_Q3_SUMMARY = (
    PROJECT_ROOT / "outputs" / "q3" / "07_最终方案摘要.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "q3-audit"
FINE_RING_RANGES = (
    (1, 1),
    (2, 3),
    (4, 5),
    (6, 8),
    (9, 11),
    (12, 14),
    (15, 17),
    (18, 20),
    (21, 24),
    (25, 28),
)


@dataclass(frozen=True)
class FineGroupDesign:
    scales: tuple[float, ...]
    heights: tuple[float, ...]


@dataclass(frozen=True)
class AuditStep:
    sweep: int
    action: str
    power_mw: float
    unit_area_power_kw_m2: float
    scales: tuple[float, ...]
    heights: tuple[float, ...]


def _fine_group_indices(mother: CampoMotherField) -> np.ndarray:
    result = np.full(mother.mirror_count, -1, dtype=np.int64)
    for group, (start, stop) in enumerate(FINE_RING_RANGES):
        mask = (
            (mother.ring_indices >= start)
            & (mother.ring_indices <= stop)
        )
        result[mask] = group
    if np.any(result < 0):
        raise ValueError("十组圆环映射没有覆盖全部镜子。")
    return result


def _load_initial_design(
    mother: CampoMotherField,
    summary_path: Path,
    fine_groups: np.ndarray,
) -> FineGroupDesign:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    group_payload = payload["group_design"]
    six_group = GroupDesign(
        scales=tuple(group_payload["scales"]),
        heights=tuple(group_payload["installation_heights_m"]),
    )
    scales: list[float] = []
    heights: list[float] = []
    for fine_group in range(len(FINE_RING_RANGES)):
        indices = np.flatnonzero(fine_groups == fine_group)
        source_group = int(mother.group_indices[indices[0]])
        scales.append(six_group.scales[source_group])
        heights.append(six_group.heights[source_group])
    return FineGroupDesign(tuple(scales), tuple(heights))


def _specifications(
    mother: CampoMotherField,
    fine_groups: np.ndarray,
    design: FineGroupDesign,
) -> ExpandedSpecifications:
    scales = np.asarray(design.scales, dtype=float)[fine_groups]
    installation_heights = np.asarray(
        design.heights,
        dtype=float,
    )[fine_groups]
    widths = mother.base_width * scales
    heights = mother.base_height * scales
    return ExpandedSpecifications(
        widths=widths,
        heights=heights,
        installation_heights=installation_heights,
        areas=widths * heights,
    )


def _evaluate(
    *,
    mother: CampoMotherField,
    fine_groups: np.ndarray,
    design: FineGroupDesign,
    profile: EvaluationProfile,
    cache: EvaluationCache,
) -> HeterogeneousEvaluation:
    return evaluate_specifications(
        coordinates=mother.coordinates,
        specifications=_specifications(mother, fine_groups, design),
        ring_indices=mother.ring_indices,
        group_indices=fine_groups,
        original_indices=np.arange(
            mother.mirror_count,
            dtype=np.int64,
        ),
        field_config=field_config_from_mother(mother),
        profile=profile,
        safety_epsilon=mother.parameters.safety_epsilon,
        cache=cache,
    )


def _replace(
    values: tuple[float, ...],
    index: int,
    value: float,
) -> tuple[float, ...]:
    result = list(values)
    result[index] = value
    return tuple(result)


def _candidate(
    design: FineGroupDesign,
    group: int,
    variable: str,
    delta: float,
) -> FineGroupDesign:
    if variable == "H":
        return FineGroupDesign(
            scales=design.scales,
            heights=_replace(
                design.heights,
                group,
                design.heights[group] + delta,
            ),
        )
    return FineGroupDesign(
        scales=_replace(
            design.scales,
            group,
            design.scales[group] + delta,
        ),
        heights=design.heights,
    )


def _record(
    evaluation: HeterogeneousEvaluation,
) -> dict[str, float | int | str]:
    return {
        "profile": evaluation.profile_name,
        "mirror_count": evaluation.mirror_count,
        "total_area_m2": evaluation.total_area_m2,
        "annual_power_mw": evaluation.annual_power_mw,
        "unit_area_power_kw_m2": (
            evaluation.unit_area_power_kw_m2
        ),
        "minimum_ground_clearance_m": (
            evaluation.geometry.minimum_ground_clearance_m
        ),
    }


def run(
    *,
    q2_summary: Path,
    q3_summary: Path,
    output_dir: Path,
    sweeps: int,
    target_power_mw: float,
) -> Path:
    mother = build_campo_mother_field(q2_summary)
    fine_groups = _fine_group_indices(mother)
    design = _load_initial_design(mother, q3_summary, fine_groups)
    cache = EvaluationCache()
    medium = medium_profile()
    initial = _evaluate(
        mother=mother,
        fine_groups=fine_groups,
        design=design,
        profile=medium,
        cache=cache,
    )
    current = initial
    trace: list[AuditStep] = []
    print(
        "十组初值："
        f"P={current.annual_power_mw:.6f} MW，"
        f"q={current.unit_area_power_kw_m2:.9f}",
        flush=True,
    )

    for sweep in range(sweeps):
        order = (
            range(len(FINE_RING_RANGES))
            if sweep % 2 == 0
            else reversed(range(len(FINE_RING_RANGES)))
        )
        accepted = 0
        for group in order:
            for variable, step in (("H", 0.1), ("s", 0.01)):
                candidates: list[
                    tuple[str, FineGroupDesign, HeterogeneousEvaluation]
                ] = []
                for direction in (-1.0, 1.0):
                    action = (
                        f"{variable}{group + 1}"
                        f"{direction * step:+.3f}"
                    )
                    proposal = _candidate(
                        design,
                        group,
                        variable,
                        direction * step,
                    )
                    try:
                        evaluation = _evaluate(
                            mother=mother,
                            fine_groups=fine_groups,
                            design=proposal,
                            profile=medium,
                            cache=cache,
                        )
                    except ValueError:
                        continue
                    if evaluation.annual_power_mw < target_power_mw:
                        continue
                    candidates.append((action, proposal, evaluation))
                if not candidates:
                    continue
                action, proposal, evaluation = max(
                    candidates,
                    key=lambda item: item[2].unit_area_power_kw_m2,
                )
                if (
                    evaluation.unit_area_power_kw_m2
                    <= current.unit_area_power_kw_m2 + 1e-5
                ):
                    continue
                design = proposal
                current = evaluation
                accepted += 1
                trace.append(
                    AuditStep(
                        sweep=sweep + 1,
                        action=action,
                        power_mw=current.annual_power_mw,
                        unit_area_power_kw_m2=(
                            current.unit_area_power_kw_m2
                        ),
                        scales=design.scales,
                        heights=design.heights,
                    )
                )
                print(
                    f"第{sweep + 1}轮接受 {action}："
                    f"P={current.annual_power_mw:.6f} MW，"
                    f"q={current.unit_area_power_kw_m2:.9f}",
                    flush=True,
                )
        print(
            f"第{sweep + 1}轮结束，接受 {accepted} 个动作。",
            flush=True,
        )
        if accepted == 0:
            break

    formal = _evaluate(
        mother=mother,
        fine_groups=fine_groups,
        design=design,
        profile=formal_profile(),
        cache=cache,
    )
    dense_settings = dense_profile()
    dense = _evaluate(
        mother=mother,
        fine_groups=fine_groups,
        design=design,
        profile=dense_settings,
        cache=cache,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "10组局部细化审计.json"
    group_counts = tuple(
        int(np.count_nonzero(fine_groups == group))
        for group in range(len(FINE_RING_RANGES))
    )
    payload = {
        "ring_ranges": FINE_RING_RANGES,
        "group_counts": group_counts,
        "sweeps_requested": sweeps,
        "target_power_mw": target_power_mw,
        "initial_medium": _record(initial),
        "final_medium": _record(current),
        "final_formal": _record(formal),
        "final_dense": _record(dense),
        "final_design": asdict(design),
        "trace": [asdict(step) for step in trace],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "十组正式复算："
        f"P={formal.annual_power_mw:.6f} MW，"
        f"q={formal.unit_area_power_kw_m2:.9f}",
        flush=True,
    )
    print(
        "十组加密复算："
        f"P={dense.annual_power_mw:.6f} MW，"
        f"q={dense.unit_area_power_kw_m2:.9f}",
        flush=True,
    )
    print(path, flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q2-summary",
        type=Path,
        default=DEFAULT_Q2_SUMMARY,
    )
    parser.add_argument(
        "--q3-summary",
        type=Path,
        default=DEFAULT_Q3_SUMMARY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--target-power", type=float, default=42.0)
    args = parser.parse_args()
    if args.sweeps < 1:
        raise ValueError("sweeps 必须大于等于 1。")
    run(
        q2_summary=args.q2_summary,
        q3_summary=args.q3_summary,
        output_dir=args.output,
        sweeps=args.sweeps,
        target_power_mw=args.target_power,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
