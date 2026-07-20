# 定日镜场优化设计

本项目对应 `task/A题.pdf` 的三问。题面和原始附件、正式源码、辅助工具、说明文档和计算输出彼此分开，避免把演示程序当成正式求解器。

> 第三问当前状态：原六组方案及正式结果完整保留；新的五节点径向连续
> Campo 模型已经完成代码重构和端到端烟雾测试，但三起点中精度收敛、
> 正式复算和加密验证按要求中止，尚未形成新的正式结论。现有
> `outputs/q3_continuous/` 中的旧两区直线结果不得作为五节点模型结果引用。

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
│       ├── 第三问.md             # 原六组第三问正文（连续模型正文待收口）
│       ├── 第三问公式说明.md     # 原六组第三问公式说明
│       ├── q3-technical-notes.md # 第三问实现接口、搜索与验证细节
│       └── q3_continuous/        # Campo 连续模型旧文档，待按五节点实跑结果重写
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
│   │   └── q3_continuous/        # 五节点径向连续 Campo 实现
│   ├── solve_q1.py               # 兼容命令行入口
│   ├── solve_q2.py               # 第二问兼容命令行入口
│   ├── solve_q3.py               # 原六组第三问入口
│   └── solve_q3_continuous.py    # Campo 连续参数化独立入口
├── tool/
│   └── heliostat3DApp.py         # 交互式三维展示，不作为正式结果
├── tests/                         # 几何和物理不变量检查
├── backups/
│   └── q3-six-group-20260721/     # 原六组文档、源码、测试与输出快照
└── outputs/
    ├── q1/                        # 第一问扁平交付包
    ├── q2/                        # 第二问扁平交付包
    ├── q3/                        # 原六组第三问正式输出
    └── q3_continuous/            # Campo 连续输出；正式重跑前仍是旧模型结果
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
- `docs/questions/q3_continuous/`：Campo 连续实验的独立说明目录；其中旧的
  区域—行号、方位角和单调放松内容已不再对应当前五节点实现，需在正式
  实验完成后按实际结果重写。

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

### 当前模型

连续方案固定问题二正式的 1469 面 Campo 镜位、塔位和布局参数，只优化
每面镜子的尺寸与安装高度。不恢复额外镜位，不删镜，不调整塔位、行距或
镜心坐标，也不使用方位角项和单调约束。

程序根据 Campo 圆环、区域切换和场地裁剪元数据自动选择五个径向控制节点。
当前固定问题二数据得到的控制环为第 1、4、12、14、28 环，对应控制半径约为
`134.765 m`、`170.866 m`、`274.773 m`、`302.486 m`、`515.913 m`。
五个分段线性基函数满足

$$
B_j(r_i)\ge 0,
\qquad
\sum_{j=1}^{5}B_j(r_i)=1.
$$

尺寸形状和安装高度定义为

$$
g_i=\sum_{j=1}^{5}\alpha_jB_j(r_i),
\qquad
\widetilde g_i=g_i-\frac{1}{N}\sum_{k=1}^{N}g_k,
$$

$$
s_i=\lambda\exp(\widetilde g_i),
\qquad
w_i=s_iw_0,
\qquad
h_i=s_ih_0,
$$

$$
H_i=\sum_{j=1}^{5}\beta_jB_j(r_i).
$$

模型包含五个尺寸节点、五个高度节点和一个全局面积系数。尺寸中心化消除
一个冗余自由度，因此共有 11 个参数、约 10 个有效自由度。

### 搜索与精度

三个初值分别为统一规格、旧连续结果的五节点投影和弱工程先验。每个初值
依次执行：

1. 高度节点搜索，步长为 `0.4、0.2、0.1、0.05 m`；
2. 固定 $\lambda=1$ 的尺寸形状搜索，步长为 `0.04、0.02、0.01、0.005`；
3. $\lambda$ 的 `0.005、0.001、0.0002` 三级网格搜索；
4. 尺寸、高度和 $\lambda$ 联合回扫，连续两轮中精度改善小于
   $10^{-5}$ 时停止，最多运行 8 轮。

所有参数接受均使用完整 60 个规定状态、`10×10` 阴影网格和 128 条截断
光线，并在满足 $\overline P\ge42\ \mathrm{MW}$ 的候选中严格选择 $q$
更高者。三个收敛解使用 `15×15`/256/60 m 正式复算；只对正式最优解执行
`20×20`/512、80 m 和 100 m 加密验证。

快速端到端烟雾测试：

```bash
conda run -n agent env PYTHONPATH=src \
python src/solve_q3_continuous.py \
  --smoke \
  --max-joint-cycles 2 \
  --output /tmp/q3-continuous-smoke
```

烟雾测试只验证五节点展开、三起点搜索、几何约束和结果导出链路，其单时刻
数值不能作为年平均结果。

正式独立运行：

```bash
conda run -n agent env PYTHONPATH=src \
python src/solve_q3_continuous.py
```

正式命令会自动完成三起点中精度搜索、三个收敛解的正式复算，以及正式最优
解的两档加密验证。运行成功后再生成单文件完整代码：

```bash
conda run -n agent python tool/build_q3_continuous_bundle.py
```

目标输出为：

```text
outputs/q3_continuous/
├── 01_完整代码.py
├── 02_逐镜最终参数.csv
├── 03_逐时刻结果.csv
├── 04_月平均结果.csv
├── 05_年平均结果.json
├── 06_搜索轨迹.csv
├── 07_最终方案摘要.json
├── 08_正式精度验证.json
├── 09_加密精度验证.json
└── 10_连续规格节点.csv
```

只运行 Campo 专项检查：

```bash
conda run -n agent env PYTHONPATH=src \
python -m unittest discover -s tests -p 'test_q3_continuous.py' -v
```

当前 8 项专项检查已通过，三起点正式实验则在运行过程中按要求中止，因此
尚无五节点模型的正式年平均结果，也不能据此判断其最终效果是否优于六组
方案。旧的 `0.687936 kW/m²` 来自已废弃的两区直线连续模型，不是本模型
最终结果。

## 三维工具

```bash
python tool/heliostat3DApp.py
```

该工具目前只计算余弦效率、大气透射率和反射率，显示的功率没有加入阴影遮挡和截断损失；论文和结果表应以对应的正式求解器及 `outputs/q*/` 为准。
