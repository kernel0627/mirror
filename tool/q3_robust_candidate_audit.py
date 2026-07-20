"""评估十组细化解的离地稳健化及小幅面积补偿候选。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from heliostat.config import SolverConfig
from heliostat.q2.evaluate import EvaluationProfile
from heliostat.q3.evaluate import (
    EvaluationCache,
    dense_profile,
    formal_profile,
)

from q3_fine_group_audit import (
    DEFAULT_Q2_SUMMARY,
    DEFAULT_OUTPUT,
    FineGroupDesign,
    _evaluate,
    _fine_group_indices,
    _record,
    _replace,
    build_campo_mother_field,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINE_AUDIT = DEFAULT_OUTPUT / "10组局部细化审计.json"


def _profile(
    *,
    name: str,
    grid: int,
    rays: int,
    radius: float,
) -> EvaluationProfile:
    return EvaluationProfile(
        name=name,
        solver=SolverConfig(
            shadow_grid_size=grid,
            truncation_rays=rays,
            neighbor_radius_m=radius,
            truncation_chunk_size=128,
            sobol_seed=2023,
        ),
    )


def _load_design(path: Path) -> FineGroupDesign:
    payload = json.loads(path.read_text(encoding="utf-8"))
    design = payload["final_design"]
    return FineGroupDesign(
        scales=tuple(design["scales"]),
        heights=tuple(design["heights"]),
    )


def _with_ground_clearance(
    design: FineGroupDesign,
    *,
    base_height: float,
    clearance_m: float,
) -> FineGroupDesign:
    heights = tuple(
        max(height, 0.5 * base_height * scale + clearance_m)
        for scale, height in zip(design.scales, design.heights)
    )
    return FineGroupDesign(scales=design.scales, heights=heights)


def run(
    *,
    q2_summary: Path,
    fine_audit: Path,
    output_dir: Path,
    clearance_m: float,
    scale_step: float,
) -> Path:
    mother = build_campo_mother_field(q2_summary)
    fine_groups = _fine_group_indices(mother)
    cache = EvaluationCache()
    original = _load_design(fine_audit)
    robust = _with_ground_clearance(
        original,
        base_height=mother.base_height,
        clearance_m=clearance_m,
    )

    formal = formal_profile()
    dense = dense_profile()
    robust_formal = _evaluate(
        mother=mother,
        fine_groups=fine_groups,
        design=robust,
        profile=formal,
        cache=cache,
    )
    robust_dense = _evaluate(
        mother=mother,
        fine_groups=fine_groups,
        design=robust,
        profile=dense,
        cache=cache,
    )
    print(
        "稳健十组基线："
        f"formal P={robust_formal.annual_power_mw:.6f}, "
        f"q={robust_formal.unit_area_power_kw_m2:.9f}; "
        f"dense P={robust_dense.annual_power_mw:.6f}, "
        f"q={robust_dense.unit_area_power_kw_m2:.9f}",
        flush=True,
    )

    candidate_rows = []
    for group in range(len(robust.scales)):
        candidate = FineGroupDesign(
            scales=_replace(
                robust.scales,
                group,
                robust.scales[group] + scale_step,
            ),
            heights=robust.heights,
        )
        candidate = _with_ground_clearance(
            candidate,
            base_height=mother.base_height,
            clearance_m=clearance_m,
        )
        evaluation = _evaluate(
            mother=mother,
            fine_groups=fine_groups,
            design=candidate,
            profile=dense,
            cache=cache,
        )
        row = {
            "group": group + 1,
            "scale_delta": scale_step,
            **_record(evaluation),
            "delta_power_mw": (
                evaluation.annual_power_mw
                - robust_dense.annual_power_mw
            ),
            "delta_area_m2": (
                evaluation.total_area_m2
                - robust_dense.total_area_m2
            ),
            "delta_q_kw_m2": (
                evaluation.unit_area_power_kw_m2
                - robust_dense.unit_area_power_kw_m2
            ),
        }
        candidate_rows.append(row)
        print(
            f"G{group + 1} s+{scale_step:.3f}: "
            f"P={evaluation.annual_power_mw:.6f}, "
            f"q={evaluation.unit_area_power_kw_m2:.9f}",
            flush=True,
        )

    high_profiles = (
        _profile(
            name="q3-dense-25x25-1024",
            grid=25,
            rays=1024,
            radius=100.0,
        ),
        _profile(
            name="q3-dense-30x30-2048",
            grid=30,
            rays=2048,
            radius=100.0,
        ),
    )
    high_rows = []
    for profile in high_profiles:
        evaluation = _evaluate(
            mother=mother,
            fine_groups=fine_groups,
            design=robust,
            profile=profile,
            cache=cache,
        )
        high_rows.append(_record(evaluation))
        print(
            f"{profile.name}: "
            f"P={evaluation.annual_power_mw:.6f}, "
            f"q={evaluation.unit_area_power_kw_m2:.9f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "10组稳健候选审计.json"
    payload = {
        "clearance_target_m": clearance_m,
        "scale_probe_step": scale_step,
        "source_design": asdict(original),
        "robust_design": asdict(robust),
        "robust_formal": _record(robust_formal),
        "robust_dense": _record(robust_dense),
        "scale_probe_dense": candidate_rows,
        "robust_high_precision": high_rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
        "--fine-audit",
        type=Path,
        default=DEFAULT_FINE_AUDIT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--clearance", type=float, default=0.1)
    parser.add_argument("--scale-step", type=float, default=0.002)
    args = parser.parse_args()
    run(
        q2_summary=args.q2_summary,
        fine_audit=args.fine_audit,
        output_dir=args.output,
        clearance_m=args.clearance,
        scale_step=args.scale_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
