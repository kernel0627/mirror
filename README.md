# 定日镜场优化设计

本项目对应 `task/A题.pdf` 的三问。题面和原始附件、正式源码、辅助工具、说明文档和计算输出彼此分开，避免把演示程序当成正式求解器。

## 目录

```text
.
├── task/                         # 题面、坐标附件和第2/3问提交模板
├── docs/
│   ├── WORK_BREAKDOWN.md         # 三问的工作边界和实现顺序
│   └── questions/
│       ├── q1-plan.md            # 第一问简版实施规格
│       ├── q1-technical-notes.md # 第一问详细推导和数值说明
│       └── q1-validation.md      # 第一问当前结果与收敛检查
├── src/
│   ├── heliostat/
│   │   ├── solar.py              # 三问共用：太阳位置和 DNI
│   │   ├── geometry.py           # 三问共用：镜场几何与姿态
│   │   ├── shadow.py             # 三问共用：阴影遮挡
│   │   ├── truncation.py         # 三问共用：截断效率
│   │   └── q1/                   # 第一问专用流程
│   │       ├── solve.py          # 逐时刻计算与命令行
│   │       ├── aggregate.py      # 月平均、年平均
│   │       ├── export.py         # 结果和论文表格输出
│   │       └── plot.py           # 两张正式结果图
│   └── solve_q1.py               # 兼容命令行入口
├── tool/
│   └── heliostat3DApp.py         # 交互式三维展示，不作为正式结果
├── tests/                         # 几何和物理不变量检查
└── outputs/
    └── q1/                        # 第一问的正式输出
```

## 环境安装

项目依赖统一记录在 `requirements.txt` 中。建议使用 Python 3.10 或更高版本，并在项目根目录创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 第一问运行

默认使用文档确定的正式精度：阴影网格 `15×15`、截断光线 `256`、邻镜半径 `60 m`。

```bash
python src/solve_q1.py
```

快速调试一个时刻：

```bash
python src/solve_q1.py \
  --months 6 \
  --times 12 \
  --limit-mirrors 20 \
  --shadow-grid 3 \
  --truncation-rays 8 \
  --output /tmp/q1-smoke
```

运行基础检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

正式结果见 `outputs/q1/`。该目录是可直接分享的扁平展示包，按编号依次包含单文件完整代码、逐时刻/月平均/年平均结果、1745 面镜子的年平均结果、运行配置、论文表格和两张正式结果图。

两张图分别展示月平均综合光学效率与单位面积输出热功率，以及单镜年平均综合光学效率的空间分布。运行收敛验证：

```bash
python src/solve_q1.py --run-validation
```

程序只保留逐时刻的全场汇总和单镜年平均结果，不保存全部单镜逐时刻数据。工程内部仍按职责拆分在 `src/heliostat/q1/`，展示包则合并为一个 `01_第一问完整代码.py`。

## 三维工具

```bash
python tool/heliostat3DApp.py
```

该工具目前只计算余弦效率、大气透射率和反射率，显示的功率没有加入阴影遮挡和截断损失；论文和结果表应以 `src/solve_q1.py` 为准。
