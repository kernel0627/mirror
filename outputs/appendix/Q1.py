"""第一问：给定镜场的逐时刻、月平均和年平均评价。"""
from __future__ import annotations
# ruff: noqa
from Public import *
from pathlib import Path
from dataclasses import asdict
import json
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'没有可写入 {path.name} 的结果。')
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _display_source_path(source_path: str | Path) -> str:
    path = Path(source_path)
    if not path.is_absolute():
        return path.as_posix()
    if 'task' in path.parts:
        task_index = path.parts.index('task')
        return Path(*path.parts[task_index:]).as_posix()
    return path.name

def write_question1_results(output_dir: str | Path, time_records: Iterable[Any], monthly_records: Iterable[Any], annual_record: Any, mirror_annual_records: Iterable[Any], field_config: FieldConfig, solver_config: SolverConfig, source_path: str | Path, mirror_count: int) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    time_rows = [asdict(record) for record in time_records]
    monthly_rows = [asdict(record) for record in monthly_records]
    annual_row = asdict(annual_record)
    mirror_annual_rows = [asdict(record) for record in mirror_annual_records]
    months = sorted({row['month'] for row in time_rows})
    solar_times = sorted({row['solar_time'] for row in time_rows})
    time_path = destination / '02_逐时刻计算结果.csv'
    monthly_path = destination / '03_月平均计算结果.csv'
    annual_path = destination / '04_年平均计算结果.json'
    mirror_annual_path = destination / '05_单镜年平均结果.csv'
    run_path = destination / '06_运行配置.json'
    _write_csv(time_path, time_rows)
    _write_csv(monthly_path, monthly_rows)
    _write_csv(mirror_annual_path, mirror_annual_rows)
    annual_path.write_text(json.dumps(annual_row, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    run_path.write_text(json.dumps({'source': _display_source_path(source_path), 'field': field_config.to_dict(), 'solver': solver_config.to_dict(), 'run': {'mirror_count': mirror_count, 'months': months, 'solar_times': solar_times, 'time_state_count': len(time_rows)}}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'time': time_path, 'monthly': monthly_path, 'annual': annual_path, 'mirror_annual': mirror_annual_path, 'config': run_path}

def write_paper_tables(output_dir: str | Path, monthly_records: Iterable[Any], annual_record: Any) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / '07_论文结果与验证表.md'
    monthly_lines = ['# 第一问结果与验证表', '', '本文档汇总第一问的月平均、年平均和数值收敛结果。', '', '## 表 1 每月 21 日平均光学效率及输出功率', '', '| 日期 | 平均光学效率 | 平均余弦效率 | 平均阴影遮挡效率 | 平均截断效率 | 单位面积镜面平均输出热功率 (kW/m²) |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for record in monthly_records:
        monthly_lines.append(f'| {record.month} 月 21 日 | {record.average_optical_efficiency:.6f} | {record.average_cosine_efficiency:.6f} | {record.average_shadow_blocking_efficiency:.6f} | {record.average_truncation_efficiency:.6f} | {record.unit_area_output_kw_m2:.6f} |')
    annual_lines = ['', '## 表 2 年平均光学效率及输出功率', '', '| 年平均光学效率 | 年平均余弦效率 | 年平均阴影遮挡效率 | 年平均截断效率 | 年平均输出热功率 (MW) | 单位面积镜面年平均输出热功率 (kW/m²) |', '| ---: | ---: | ---: | ---: | ---: | ---: |', f'| {annual_record.average_optical_efficiency:.6f} | {annual_record.average_cosine_efficiency:.6f} | {annual_record.average_shadow_blocking_efficiency:.6f} | {annual_record.average_truncation_efficiency:.6f} | {annual_record.field_output_mw:.6f} | {annual_record.unit_area_output_kw_m2:.6f} |']
    table_path.write_text('\n'.join(monthly_lines + annual_lines) + '\n', encoding='utf-8')
    return {'paper_tables': table_path}

def write_validation_table(output_dir: str | Path, validation_records: Iterable[Any]) -> dict[str, Path]:
    destination = Path(output_dir)
    rows = [asdict(record) for record in validation_records]
    table_path = destination / '07_论文结果与验证表.md'
    lines = ['', '## 表 3 数值收敛验证', '', '| 验证项目 | 参数 | 观测指标 | 数值 | 相对正式配置差异 | 运行时间 (s) |', '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['category']} | {row['parameter']} | {row['metric']} | {row['value']:.6f} | {row['relative_difference_percent']:.4f}% | {row['runtime_seconds']:.3f} |")
    with table_path.open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return {'validation_table': table_path}

ROOT = Path(__file__).resolve().parents[2]

def main():
    field_config = FieldConfig()
    solver_config = SolverConfig(
        shadow_grid_size=15,
        truncation_rays=256,
        neighbor_radius_m=60.0,
        sobol_seed=2023,
    )
    mirror_xy = load_mirror_xy(ROOT / "task" / "A" / "fj.xlsx")
    prepared = prepare_field(mirror_xy, field_config)
    solution = solve_question1(prepared, solver_config)
    output = ROOT / "outputs" / "q1"
    write_question1_results(
        output, solution.time_results, solution.monthly_results,
        solution.annual_result, solution.mirror_annual_results,
        field_config, solver_config, ROOT / "task" / "A" / "fj.xlsx",
        prepared.mirror_count,
    )
    write_paper_tables(output, solution.monthly_results, solution.annual_result)
    validation = run_validation_suite(prepared, solver_config)
    write_validation_table(output, validation)
    annual = solution.annual_result
    print(f"年平均输出热功率：{annual.field_output_mw:.6f} MW")
    print(f"单位面积年平均输出：{annual.unit_area_output_kw_m2:.6f} kW/m²")

if __name__ == "__main__":
    main()
