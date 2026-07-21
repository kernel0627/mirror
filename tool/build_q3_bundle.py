"""把第三问及其依赖合并为可独立运行的单文件展示稿。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "q3" / "01_第三问完整代码.py"
SOURCE_PATHS = (
    "src/heliostat/config.py",
    "src/heliostat/solar.py",
    "src/heliostat/geometry.py",
    "src/heliostat/shadow.py",
    "src/heliostat/truncation.py",
    "src/heliostat/io.py",
    "src/heliostat/q1/aggregate.py",
    "src/heliostat/q1/export.py",
    "src/heliostat/q1/plot.py",
    "src/heliostat/q1/solve.py",
    "src/heliostat/q2/layout.py",
    "src/heliostat/q2/evaluate.py",
    "src/heliostat/q3/_baseline.py",
    "src/heliostat/q3/_optics.py",
    "src/heliostat/q3/_workbook.py",
    "src/heliostat/q3/model.py",
    "src/heliostat/q3/tower_modes.py",
    "src/heliostat/q3/evaluate.py",
    "src/heliostat/q3/sensitivity.py",
    "src/heliostat/q3/search.py",
    "src/heliostat/q3/closure.py",
    "src/heliostat/q3/export.py",
    "src/heliostat/q3/plot.py",
    "src/heliostat/q3/solve.py",
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


class _BundleTransformer(ast.NodeTransformer):
    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> ast.ImportFrom | ast.Assign | list[ast.Assign] | None:
        if node.module == "__future__":
            return None
        if node.level > 0:
            aliases = [
                ast.Assign(
                    targets=[ast.Name(id=item.asname, ctx=ast.Store())],
                    value=ast.Name(id=item.name, ctx=ast.Load()),
                )
                for item in node.names
                if item.asname is not None
            ]
            return aliases or None
        return node

    def visit_If(self, node: ast.If) -> ast.If | None:
        if _is_main_guard(node):
            return None
        return self.generic_visit(node)


def _render_module(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    transformed = _BundleTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def build_bundle(output_path: Path = OUTPUT_PATH) -> Path:
    sections = [
        '"""第三问完整代码展示稿。\n\n'
        "本文件合并共享光学核心、Campo 母场、异构搜索、验证和输出流程，"
        "可直接运行。\n"
        '"""',
        "from __future__ import annotations",
        "# ruff: noqa: E402,F401,F811",
    ]
    for relative_path in SOURCE_PATHS:
        source_path = PROJECT_ROOT / relative_path
        sections.extend(
            (
                "",
                "# " + "=" * 72,
                f"# 来源：{relative_path}",
                "# " + "=" * 72,
                "",
                _render_module(source_path),
            )
        )
    sections.extend(
        (
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(run())",
            "",
        )
    )
    content = "\n".join(sections)
    content = content.replace(
        "Path(__file__).resolve().parents[3]",
        "Path(__file__).resolve().parents[2]",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(build_bundle())
