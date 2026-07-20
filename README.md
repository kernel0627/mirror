# 定日镜场优化设计

本项目对应 `task/A题.pdf` 的三问。题面和原始附件、正式源码、辅助工具、说明文档和计算输出彼此分开，避免把演示程序当成正式求解器。

## 目录

```text
.
├── task/                         # 题面、坐标附件和第2/3问提交模板
├── docs/
│   ├── WORK_BREAKDOWN.md         # 三问的工作边界和实现顺序
│   └── questions/
│       ├── 第一问.md             # 第一问正式建模与结果方案
│       ├── 第二问.md             # 第二问正式优化与结果方案
│       ├── q1-plan.md            # 第一问简版实施规格
│       ├── q1-technical-notes.md # 第一问详细推导和数值说明
│       ├── q1-validation.md      # 第一问当前结果与收敛检查
│       └── q2-technical-notes.md # 第二问完整推导与搜索细节
├── src/
│   ├── heliostat/
│   │   ├── solar.py              # 三问共用：太阳位置和 DNI
│   │   ├── geometry.py           # 三问共用：镜场几何与姿态
│   │   ├── shadow.py             # 三问共用：阴影遮挡
│   │   ├── truncation.py         # 三问共用：截断效率
│   │   ├── q1/                   # 第一问专用流程
│   │   │   ├── solve.py          # 逐时刻计算与命令行
│   │   │   ├── aggregate.py      # 月平均、年平均
│   │   │   ├── export.py         # 结果和论文表格输出
│   │   │   └── plot.py           # 两张正式结果图
│   │   └── q2/                   # 第二问双布局生成、评价与搜索
│   │       ├── layout.py         # 分区圆环和改进 Campo
│   │       ├── evaluate.py       # 复用问题一模型与外边界扫描
│   │       ├── search.py         # 多起点循环变步长搜索
│   │       ├── prune.py          # 外层对称镜位结构化修剪
│   │       ├── export.py         # 结果文件与 result2.xlsx
│   │       ├── plot.py           # 四张正式结果图
│   │       └── solve.py          # 第二问命令行流程
│   ├── solve_q1.py               # 兼容命令行入口
│   └── solve_q2.py               # 第二问兼容命令行入口
├── tool/
│   └── heliostat3DApp.py         # 交互式三维展示，不作为正式结果
├── tests/                         # 几何和物理不变量检查
└── outputs/
    ├── q1/                        # 第一问扁平交付包
    └── q2/                        # 第二问扁平交付包
```

## 建模方案文档

- `docs/questions/第一问.md`：第一问的固定镜场评价方案、正式结果、收敛验证以及与第二问的关系；
- `docs/questions/第二问.md`：正文版方案，明确同心圆简写、Campo 详写，并标注每张表和图片的数据来源；
- `docs/questions/q2-technical-notes.md`：第二问的完整布局推导与搜索细节附件。

两份文档是面向论文和交付的主方案；`q1-plan.md`、`q1-technical-notes.md` 和 `q1-validation.md` 保留为第一问的实施与推导附件。

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

## 第二问运行

第二问并行优化分区交错同心圆和改进 Campo 径向交错两种布局。两种布局分别使用 Sobol 分散初值和循环变步长搜索，统一保留 `0.01 m` 镜心距离安全余量，最终用问题一正式精度复算并比较。

先运行只含一个时刻的端到端烟雾测试：

```bash
python src/solve_q2.py \
  --smoke \
  --initial-samples 1 \
  --retained-starts 1 \
  --max-cycles 1 \
  --coarse-stride 20 \
  --extent-window 0 \
  --prune-rounds 0 \
  --output /tmp/q2-smoke
```

烟雾测试只验证布局、搜索、评价和 Excel 输出链路，其功率不是正式年平均结果。正式双布局搜索使用：

```bash
python src/solve_q2.py
```

正式运行会反复评价上千面镜子的 60 个时刻，计算量明显高于第一问。可先通过 `--initial-samples`、`--retained-starts` 和 `--max-cycles` 控制预算，再逐步扩大搜索规模。增加 `--run-validation` 可执行 `20×20` 阴影网格、512 条截断光线和 80 m 邻镜半径的加密复算。

正式结果见 `outputs/q2/`。该目录与第一问一样是可直接分享的扁平展示包：含 `01_第二问完整代码.py`、两布局比较、最终坐标、月年平均与单镜结果、论文结果与验证表、高精度复核、提交用 Excel 和四张正式图片。当前最终方案为采用 1 cm 安全余量的 1469 面改进 Campo 镜场；正式精度年平均输出为 `42.044238 MW`，加密复算为 `42.055115 MW`。

单文件展示稿也可独立运行，不依赖 `src/heliostat/`：

```bash
python outputs/q2/01_第二问完整代码.py --help
```

## 三维工具

```bash
python tool/heliostat3DApp.py
```

该工具目前只计算余弦效率、大气透射率和反射率，显示的功率没有加入阴影遮挡和截断损失；论文和结果表应以对应的正式求解器及 `outputs/q*/` 为准。
