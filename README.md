# 定日镜场优化设计

本项目对应 `task/A题.pdf` 的三问。题面和原始附件、正式源码、辅助工具、说明文档和计算输出彼此分开，避免把演示程序当成正式求解器。

> 第三问当前状态：原六组方案及正式结果完整保留，作为最终数值 baseline；
> 新的径向—角度连续 Campo2D 模型已完成代码、8 项专项测试和端到端 smoke
> 验证，但尚未执行多起点正式搜索、正式复算和加密验证。smoke 的单时刻
> 数值以及 `outputs/q3_continuous/` 中的历史结果均不得作为新模型结论引用。

## 目录

```text
.
├── task/                         # 题面、坐标附件和第2/3问提交模板
├── docs/
│   ├── WORK_BREAKDOWN.md         # 三问的工作边界和实现顺序
│   └── questions/
│       ├── 第一问.md             # 第一问正式建模与结果方案
│       ├── 第一问公式说明.md     # 第一问完整公式总表与逐式定义
│       ├── 第二问.md             # 第二问正式优化与结果方案
│       ├── 第二问公式说明.md     # 第二问完整优化、布局与选择公式
│       ├── q1-plan.md            # 第一问简版实施规格
│       ├── q1-technical-notes.md # 第一问详细推导和数值说明
│       ├── q1-validation.md      # 第一问当前结果与收敛检查
│       ├── q2-technical-notes.md # 第二问完整推导与搜索细节
│       ├── 第三问.md             # 新 Campo2D 第三问正文
│       ├── 第三问公式说明.md     # 新 Campo2D 完整公式系统
│       ├── q3-technical-notes.md # 原六组实现接口与验证细节
│       ├── q3_continuous/        # 旧五节点纯径向连续模型文档
│       └── q3_campo2d/           # 新模型运行与实现说明
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
│   │   ├── q2/                   # 第二问双布局生成、评价与搜索
│   │   │   ├── layout.py         # 分区圆环和改进 Campo
│   │   │   ├── evaluate.py       # 复用问题一模型与外边界扫描
│   │   │   ├── search.py         # 多起点循环变步长搜索
│   │   │   ├── prune.py          # 外层对称镜位结构化修剪
│   │   │   ├── export.py         # 结果文件与 result2.xlsx
│   │   │   ├── plot.py           # 四张正式结果图
│   │   │   └── solve.py          # 第二问命令行流程
│   │   ├── q3/                   # 原六组第三问实现
│   │   │   ├── model.py          # 1471 面母场、六组规格和异构几何
│   │   │   ├── evaluate.py       # 异构评价、精度配置和经验校准
│   │   │   ├── search.py         # 高度、面积再分配和面积压缩搜索
│   │   │   ├── prune.py          # 低贡献对称镜位结构化删镜
│   │   │   ├── export.py         # 第三问结果和 result3.xlsx
│   │   │   └── solve.py          # 第三问命令行流程
│   │   ├── q3_continuous/        # 旧五节点纯径向连续 Campo 实现
│   │   └── q3_campo2d/           # 新径向—角度连续 Campo 实现
│   ├── solve_q1.py               # 兼容命令行入口
│   ├── solve_q2.py               # 第二问兼容命令行入口
│   ├── solve_q3.py               # 原六组第三问入口
│   ├── solve_q3_continuous.py    # 旧纯径向连续模型入口
│   └── solve_q3_campo2d.py       # 新 Campo2D 正式入口
├── tool/
│   └── heliostat3DApp.py         # 交互式三维展示，不作为正式结果
├── tests/                         # 几何和物理不变量检查
├── backups/
│   └── q3-six-group-20260721/     # 原六组文档、源码、测试与输出快照
└── outputs/
    ├── q1/                        # 第一问扁平交付包
    ├── q2/                        # 第二问扁平交付包
    ├── q3/                        # 原六组第三问正式输出
    ├── q3_continuous/             # 历史连续模型结果，不作为新结论
    └── q3_campo2d/                # 新模型正式运行后生成
```

## 建模方案文档

- `docs/questions/第一问.md`：第一问的固定镜场评价方案、正式结果、收敛验证以及与第二问的关系；
- `docs/questions/第一问公式说明.md`：第一问从太阳位置、四项效率到月年汇总的完整公式系统；
- `docs/questions/第二问.md`：正文版方案，明确同心圆简写、Campo 详写，并标注每张表和图片的数据来源；
- `docs/questions/第二问公式说明.md`：第二问目标、约束、双布局生成、搜索、修剪和方案选择的完整公式系统；
- `docs/questions/q2-technical-notes.md`：第二问的完整布局推导与搜索细节附件。
- `docs/questions/第三问.md`：塔位和 Campo 几何微调下的径向—角度连续异构优化正文；
- `docs/questions/第三问公式说明.md`：新模型的 Campo 重建、连续规格、异构评价、约束与搜索公式；
- `docs/questions/q3_campo2d/技术说明.md`：新实现的独立边界、文件职责、运行命令和结果状态；
- `docs/questions/q3-technical-notes.md`：原六组方案的实现接口和验证细节；
- `docs/questions/q3_continuous/`：旧五节点纯径向连续实验的历史说明。

三份中文题目文档是面向论文和交付的正文方案；对应“公式说明”负责集中列全公式，其他技术说明保留为实施、推导、搜索与验证附件。

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

## 第三问运行

### 当前主线：径向—角度连续 Campo2D

第三问以问题二的 1469 面改进 Campo 镜场为中心初值，同时优化塔的南北
坐标、初始径向行距、行距增长量、外层圆环前缀、五节点径向尺寸与高度、
同环中心化角度修正以及全局面积尺度。几何变量变化后会重新生成 Campo，
不会把原镜位固定后只移动塔。

Sobol 序列只负责产生分散初值；局部优化采用整批 best-improvement：先用
粗精度统一排序候选，再把前若干个候选送入中精度复算。候选只有在中精度
满足 $\overline P\ge42\ \mathrm{MW}$ 且单位面积输出 $q$ 改善时才会被接受。

先运行 8 项专项检查：

```bash
conda run -n agent env PYTHONPATH=src \
python -m unittest discover -s tests -p 'test_q3_campo2d.py' -v
```

端到端 smoke 使用一个规定时刻和最小搜索预算：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/codex-matplotlib PYTHONPATH=src \
python src/solve_q3_campo2d.py \
  --smoke \
  --sobol-count 1 \
  --retained-starts 2 \
  --max-rounds 1 \
  --max-joint-cycles 1 \
  --medium-candidates 1 \
  --output /tmp/q3-campo2d-smoke
```

smoke 只验证动态 Campo 重建、Sobol 起点、分块搜索、异构评价、Excel 和
图片输出链路。它只有六月正午一个状态，任何功率、面积或 $q$ 数值都不是
正式年平均结果。

正式多起点搜索使用：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/codex-matplotlib PYTHONPATH=src \
python src/solve_q3_campo2d.py
```

正式命令默认生成 16 个 Sobol 起点、保留 3 个起点，完成中精度收敛、三个
候选的正式复算以及最终解的 80 m、100 m 加密验证。正式运行成功后生成
单文件展示稿：

```bash
conda run -n agent python tool/build_q3_campo2d_bundle.py
```

正式输出目录为 `outputs/q3_campo2d/`，按 01–20 编号保存完整代码、初值、
搜索轨迹、逐镜结果、逐时刻/月/年结果、逐环与角度统计、baseline 比较、
几何与加密验证、`result3.xlsx`、论文表格以及四张正式图片。

当前新模型的代码、8 项专项测试和一次最小 smoke 已通过；正式搜索尚未运行，
因此还不能判断它是否优于六组方案。完整方案见
`docs/questions/第三问.md`，公式见 `docs/questions/第三问公式说明.md`。

### 六组正式 baseline 与历史连续模型

原六组源码、测试和正式输出仍完整保留在 `src/heliostat/q3/`、
`tests/test_q3.py` 和 `outputs/q3/`，额外快照位于
`backups/q3-six-group-20260721/`。六组正式结果为 1471 面镜子、
总镜面面积 `60777.391 m²`、年平均输出 `42.051608 MW`、单位面积输出
`0.691896 kW/m²`；它只用于新模型完成后的统一精度比较，不用于新模型
初值或候选接受。

`src/heliostat/q3_continuous/`、`src/solve_q3_continuous.py`、
`docs/questions/q3_continuous/` 和 `outputs/q3_continuous/` 属于旧五节点纯
径向连续实验。旧的 `0.687936 kW/m²` 不代表 Campo2D 新模型结果。

## 三维工具

```bash
python tool/heliostat3DApp.py
```

该工具目前只计算余弦效率、大气透射率和反射率，显示的功率没有加入阴影遮挡和截断损失；论文和结果表应以对应的正式求解器及 `outputs/q*/` 为准。
