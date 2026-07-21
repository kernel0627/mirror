"""把第三问 Campo2D 实现及依赖合并为独立单文件展示稿。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "q3_campo2d" / "01_第三问完整代码.py"
SOURCE_PATHS = (
    "src/heliostat/config.py",
    "src/heliostat/solar.py",
    "src/heliostat/geometry.py",
    "src/heliostat/shadow.py",
    "src/heliostat/truncation.py",
    "src/heliostat/q1/aggregate.py",
    "src/heliostat/q1/solve.py",
    "src/heliostat/q2/layout.py",
    "src/heliostat/q2/evaluate.py",
    "src/heliostat/q3_campo2d/model.py",
    "src/heliostat/q3_campo2d/evaluate.py",
    "src/heliostat/q3_campo2d/search.py",
    "src/heliostat/q3_campo2d/export.py",
    "src/heliostat/q3_campo2d/plot.py",
    "src/heliostat/q3_campo2d/solve.py",
)


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class _Transformer(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module == "__future__" or node.level > 0:
            return None
        return node

    def visit_If(self, node: ast.If) -> ast.If | None:
        if _is_main_guard(node):
            return None
        return self.generic_visit(node)


def _render(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    transformed = _Transformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def build_bundle(output_path: Path = OUTPUT_PATH) -> Path:
    sections = [
        '"""第三问径向—角度连续 Campo 完整代码。"""',
        "from __future__ import annotations",
        "# ruff: noqa: E402,F401,F811",
    ]
    for relative in SOURCE_PATHS:
        sections.extend(
            (
                "",
                "# " + "=" * 72,
                f"# 来源：{relative}",
                "# " + "=" * 72,
                "",
                _render(PROJECT_ROOT / relative),
            )
        )
    sections.extend(("", 'if __name__ == "__main__":', "    raise SystemExit(run())", ""))
    content = "\n".join(sections).replace(
        "Path(__file__).resolve().parents[3]",
        "Path(__file__).resolve().parents[2]",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(build_bundle())
