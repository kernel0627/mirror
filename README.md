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
│       ├── 第一问公式说明.md     # 第一问完整公式总表与逐式定义
│       ├── 第二问.md             # 第二问正式优化与结果方案
│       ├── 第二问公式说明.md     # 第二问完整优化、布局与选择公式
│       ├── q1-plan.md            # 第一问简版实施规格
│       ├── q1-technical-notes.md # 第一问详细推导和数值说明
│       ├── q1-validation.md      # 第一问当前结果与收敛检查
│       ├── q2-technical-notes.md # 第二问完整推导与搜索细节
│       ├── 第三问.md             # 第三问分组异构优化正文方案
│       ├── 第三问公式说明.md     # 第三问逐镜异构模型与优化公式
│       ├── q3-technical-notes.md # 第三问实现接口、搜索与验证细节
│       └── q3_continuous/        # 独立 Campo 连续参数化文档
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
│   │   └── q3_continuous/        # 独立 Campo 连续参数化实现
│   ├── solve_q1.py               # 兼容命令行入口
│   ├── solve_q2.py               # 第二问兼容命令行入口
│   ├── solve_q3.py               # 原六组第三问入口
│   └── solve_q3_continuous.py    # Campo 连续参数化独立入口
├── tool/
│   └── heliostat3DApp.py         # 交互式三维展示，不作为正式结果
├── tests/                         # 几何和物理不变量检查
└── outputs/
    ├── q1/                        # 第一问扁平交付包
    ├── q2/                        # 第二问扁平交付包
    ├── q3/                        # 原六组第三问正式输出
    └── q3_continuous/            # Campo 连续参数化独立输出
```

## 建模方案文档

- `docs/questions/第一问.md`：第一问的固定镜场评价方案、正式结果、收敛验证以及与第二问的关系；
- `docs/questions/第一问公式说明.md`：第一问从太阳位置、四项效率到月年汇总的完整公式系统；
- `docs/questions/第二问.md`：正文版方案，明确同心圆简写、Campo 详写，并标注每张表和图片的数据来源；
- `docs/questions/第二问公式说明.md`：第二问目标、约束、双布局生成、搜索、修剪和方案选择的完整公式系统；
- `docs/questions/q2-technical-notes.md`：第二问的完整布局推导与搜索细节附件。
- `docs/questions/第三问.md`：固定问题二 Campo 几何结构下的六组异构规格优化正文方案；
- `docs/questions/第三问公式说明.md`：逐镜宽、高、安装高度、面积加权、经验校准和删镜公式；
- `docs/questions/q3-technical-notes.md`：异构核心接口、六组映射、搜索动作和验证流程。
- `docs/questions/q3_continuous/`：不覆盖原六组方案的 Campo 区域—行号连续
  参数化正文、公式和独立实现说明。

三份中文题目文档是面向论文和交付的正文方案；对应“公式说明”负责集中列全公式，`q1-plan.md`、`q1-technical-notes.md`、`q1-validation.md`、`q2-technical-notes.md` 和 `q3-technical-notes.md` 保留为实施、推导、搜索与验证附件。

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

第三问固定问题二最终 Campo 塔位和镜心平面坐标，恢复最后删除的对称镜位，
构造 1471 面完整母场，再对六个径向结构组联合优化镜面尺度和安装高度。

先运行一个规定时刻的端到端烟雾测试：

```bash
python src/solve_q3.py \
  --smoke \
  --calibration-candidates 1 \
  --max-cycles 1 \
  --prune-rounds 0 \
  --output /tmp/q3-smoke
```

烟雾测试验证异构评价、搜索和 `result3.xlsx` 输出链路，不能作为正式结果。
正式搜索使用：

```bash
python src/solve_q3.py
```

增加 `--run-validation` 后，会对最终正式候选执行 `20×20` 阴影网格、
512 条截断光线以及 80 m、100 m 邻镜半径的加密与敏感性复算。第三问
正式运行需要多次评价完整 1471 面镜场，计算量高于单次问题二复算。

当前正式方案保留 1471 面镜子，总镜面面积为 `60777.391 m²`，正式精度
年平均输出为 `42.051608 MW`，单位面积输出为 `0.691896 kW/m²`；
相对问题二最终方案提高约 `1.590%`。80 m 加密复算为
`42.031084 MW`，扩大到 100 m 邻域后结果不变。正式交付文件见
`outputs/q3/`。

## 第三问 Campo 连续参数化独立实验

原六组第三问继续保留在 `src/heliostat/q3/`、
`docs/questions/第三问*.md` 和 `outputs/q3/`；额外快照位于
`backups/q3-six-group-20260721/`。

连续方案不人工切分六组，直接使用 Campo 区域、区内行号和可选同环方位
特征。独立运行：

```bash
conda run -n agent env PYTHONPATH=src \
python src/solve_q3_continuous.py
```

正式结果保留 1469 面镜子，总面积为 `61300.700 m²`，年平均输出为
`42.170954 MW`，单位面积输出为 `0.687936 kW/m²`。加密复算为
`42.178822 MW` 和 `0.688064 kW/m²`。该结果优于问题二统一规格，但当前
预算下仍低于原六组方案的 `0.691896 kW/m²`。独立输出位于
`outputs/q3_continuous/`。

## 三维工具

```bash
python tool/heliostat3DApp.py
```

该工具目前只计算余弦效率、大气透射率和反射率，显示的功率没有加入阴影遮挡和截断损失；论文和结果表应以对应的正式求解器及 `outputs/q*/` 为准。
