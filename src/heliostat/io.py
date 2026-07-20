"""三问共用的坐标输入。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openpyxl import load_workbook


FloatArray = NDArray[np.float64]


def load_mirror_xy(path: str | Path, expected_count: int | None = 1745) -> FloatArray:
    """从题目附件读取定日镜 x、y 坐标。"""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到定日镜坐标文件：{source}")

    if source.suffix.lower() == ".xlsx":
        workbook = load_workbook(source, read_only=True, data_only=True)
        sheet = workbook.active
        rows = list(sheet.iter_rows(min_row=2, values_only=True))
        workbook.close()
        values = [(row[0], row[1]) for row in rows if row[0] is not None]
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            values = [(row[0], row[1]) for row in reader if row]
    else:
        raise ValueError("坐标文件只支持 .xlsx 或 .csv。")

    try:
        mirror_xy = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"坐标文件包含非数值数据：{source}") from exc

    if mirror_xy.ndim != 2 or mirror_xy.shape[1] != 2:
        raise ValueError(f"坐标数据应为 N×2，实际形状为 {mirror_xy.shape}。")
    if not np.all(np.isfinite(mirror_xy)):
        raise ValueError("坐标数据包含 NaN 或无穷值。")
    if expected_count is not None and mirror_xy.shape[0] != expected_count:
        raise ValueError(
            f"应读取 {expected_count} 面定日镜，实际读取 {mirror_xy.shape[0]} 面。"
        )
    return mirror_xy
