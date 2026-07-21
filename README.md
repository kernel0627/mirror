# 定日镜场优化设计

本仓库给出 2023 年全国大学生数学建模竞赛 A 题三问的完整计算实现，包括题目
输入、共用光学模型、三问求解程序、模型与公式说明、自动测试、结果图表和题目
要求的 Excel 提交文件。

三问使用同一套太阳位置、镜面姿态、阴影遮挡、大气透射和截断效率模型：第一问
评价给定镜场，第二问优化统一规格镜场，第三问在第二问 Campo 结构上继续优化
不同区域的镜面规格。

## 三问内容与主要结果

- 第一问读取题目给定的 1745 个镜位，计算 12 个月、每月 5 个规定时刻下的
  光学效率与输出热功率。年平均输出热功率为
  $40.430243\ \mathrm{MW}$，单位镜面面积年平均输出为
  $0.602737\ \mathrm{kW/m^2}$。
- 第二问分别优化分区交错同心圆和改进 Campo 两种统一规格布局。最终采用
  1469 面改进 Campo 镜场。年平均功率下限为 $42\ \mathrm{MW}$，实际年平均
  输出热功率为 $42.044238\ \mathrm{MW}$，功率余量为
  $0.044238\ \mathrm{MW}$，单位镜面面积年平均输出为
  $0.681068\ \mathrm{kW/m^2}$。
- 第三问在 1471 面 Campo 镜场上采用六区异构规格，完成塔位与几何参数扫描、
  规格敏感性筛选、局部搜索、正式收口、加密复算和分区边界检验。最终总采光
  面积为 $60719.432482\ \mathrm{m^2}$，正式年平均输出热功率为
  $42.086214740\ \mathrm{MW}$，相对 $42\ \mathrm{MW}$ 下限的余量为
  $0.086214740\ \mathrm{MW}$，单位镜面面积年平均输出为
  $0.693125957\ \mathrm{kW/m^2}$。加密复算的年平均输出热功率为
  $42.076085938\ \mathrm{MW}$，单位镜面面积年平均输出为
  $0.692959144\ \mathrm{kW/m^2}$。

以上结果均来自完整 60 个规定状态的正式计算。快速检查模式只用于验证程序能否
从输入运行到输出，不作为数值结论。

## 完整目录

```text
.
├── .gitignore                       # Python 缓存、虚拟环境、编辑器和临时文件规则
├── README.md                         # 项目总览、安装、运行和结果说明
├── requirements.txt                 # 计算、绘图、Excel 和三维工具依赖
├── task/                            # 题面、输入数据和提交模板
│   ├── README.md                    # task 目录文件说明
│   ├── A题.pdf                      # 2023 年国赛 A 题题面
│   ├── A/
│   │   ├── fj.xlsx                  # 第一问给定的 1745 个镜位坐标
│   │   ├── result2.xlsx             # 第二问题目提交模板
│   │   └── result3.xlsx             # 第三问题目提交模板
│   └── templates/
│       ├── 论文模版.docx            # 论文排版模板
│       └── 论文模版.json            # 模板结构化解析结果
├── src/                             # 可维护的正式源码
│   ├── solve_q1.py                  # 第一问命令行入口
│   ├── solve_q2.py                  # 第二问命令行入口
│   ├── solve_q3.py                  # 第三问命令行入口
│   └── heliostat/
│       ├── __init__.py
│       ├── config.py                # 场地、集热器、镜面与数值精度配置
│       ├── io.py                    # 镜位坐标读取与输入检查
│       ├── solar.py                 # 太阳赤纬、时角、太阳方向和 DNI
│       ├── geometry.py              # 镜面姿态、反射方向和共用几何量
│       ├── shadow.py                # 网格射线追踪与阴影遮挡效率
│       ├── truncation.py            # Sobol 太阳锥采样与截断效率
│       ├── q1/
│       │   ├── __init__.py
│       │   ├── solve.py             # 逐时刻评价、正式求解和验证入口
│       │   ├── aggregate.py         # 月平均、年平均和单镜年平均汇总
│       │   ├── export.py            # CSV、JSON 与验证表导出
│       │   └── plot.py              # 第一问两张结果图
│       ├── q2/
│       │   ├── __init__.py
│       │   ├── layout.py            # 两种布局生成与统一几何检查
│       │   ├── evaluate.py          # 候选评价、缓存和外边界扫描
│       │   ├── search.py            # Sobol 初值与循环变步长搜索
│       │   ├── prune.py             # 最外层东西对称镜位修剪
│       │   ├── solve.py             # 双布局优化、复算和验收主流程
│       │   ├── export.py            # 结果文件与 result2.xlsx 导出
│       │   └── plot.py              # 第二问四张结果图与绘图数据
│       └── q3/
│           ├── __init__.py
│           ├── six_group_baseline.json # 原六区严格回归基准
│           ├── _baseline.py         # Campo 母场、六区展开和异构几何检查
│           ├── _optics.py           # 异构镜场光学评价、缓存与精度配置
│           ├── _workbook.py         # result3.xlsx 模板写入
│           ├── model.py             # 21 维设计对象与逐镜规格展开
│           ├── tower_modes.py       # 塔位模式 A/B
│           ├── evaluate.py          # 候选预检与四级精度评价
│           ├── sensitivity.py       # 规格敏感性与六区边界扰动
│           ├── search.py            # 活跃变量分块局部搜索
│           ├── closure.py           # 塔位包围、细扫与最细邻域检查
│           ├── solve.py             # 第三问完整分阶段主流程
│           ├── export.py            # 数据、验证表和提交文件导出
│           └── plot.py              # 第三问五张结果图
├── docs/
│   ├── WORK_BREAKDOWN.md            # 计算链路、模块职责和变更影响说明
│   └── questions/
│       ├── 第一问.md                # 第一问建模路线、结果和结论
│       ├── 第一问公式说明.md        # 第一问完整公式体系
│       ├── q1-plan.md               # 第一问简化实施规格
│       ├── q1-technical-notes.md    # 第一问射线求交和实现细节
│       ├── q1-validation.md         # 第一问收敛与回归验证
│       ├── 第二问.md                # 第二问两种布局、结果和验证
│       ├── 第二问公式说明.md        # 第二问优化与布局公式
│       ├── q2-technical-notes.md    # 第二问搜索和实现细节
│       ├── 第三问.md                # 第三问优化、收口与边界检验
│       ├── 第三问公式说明.md        # 第三问目标、约束和搜索公式
│       └── q3-technical-notes.md    # 第三问代码、预算和输出字段
├── tests/
│   ├── test_core.py                 # 共用太阳、反射、阴影和截断测试
│   ├── test_q1.py                   # 第一问汇总与导出测试
│   ├── test_q2.py                   # 第二问布局、评价与提交文件测试
│   └── test_q3.py                   # 第三问回归、收口、边界和提交测试
├── tool/
│   ├── build_q3_bundle.py           # 重建第三问可独立运行的单文件程序
│   └── heliostat3DApp.py            # 镜场三维交互查看工具
└── outputs/                         # 完整计算生成的正式结果
    ├── README.md                    # 所有输出文件的逐项说明
    ├── q1/
    │   ├── 01_第一问完整代码.py
    │   ├── 02_逐时刻计算结果.csv
    │   ├── 03_月平均计算结果.csv
    │   ├── 04_年平均计算结果.json
    │   ├── 05_单镜年平均结果.csv
    │   ├── 06_运行配置.json
    │   ├── 07_论文结果与验证表.md
    │   ├── 08_月平均光学性能与输出热功率.png
    │   └── 09_单镜年平均综合光学效率空间分布.png
    ├── q2/
    │   ├── 01_第二问完整代码.py
    │   ├── 02_双布局比较.json
    │   ├── 03_最终镜位坐标.csv
    │   ├── 04_月平均计算结果.csv
    │   ├── 05_年平均计算结果.json
    │   ├── 06_单镜年平均结果.csv
    │   ├── 07_最终方案摘要.json
    │   ├── 08_论文结果与验证表.md
    │   ├── 09_高精度加密验证.json
    │   ├── 11_图2-1_两种候选布局平面分布与单镜年平均输出.png
    │   ├── 12_图2-2_两种候选布局主要性能指标对比.png
    │   ├── 13_图2-3_两种候选布局月平均性能对比.png
    │   ├── 14_图2-4_两种候选布局三维镜场与代表性中心光路.png
    │   ├── 15_双布局月平均对比数据.csv
    │   └── result2.xlsx
    └── q3/
        ├── 01_第三问完整代码.py
        ├── 02_六组回归结果.json
        ├── 03_塔位两种语义扫描.csv
        ├── 04_Campo几何粗扫.csv
        ├── 05_规格参数敏感性.csv
        ├── 06_活跃变量集合.json
        ├── 07_局部搜索轨迹.csv
        ├── 08_正式候选比较.csv
        ├── 09_最终六区参数.csv
        ├── 10_最终逐镜参数与坐标.csv
        ├── 11_正式结果比较.json
        ├── 12_加密验收比较.json
        ├── 13_几何约束验证.json
        ├── 14_局部收口检查.csv
        ├── 15_论文结果与验证表.md
        ├── 16_参数敏感性图.png
        ├── 17_六区宽高与安装高度图.png
        ├── 18_六组与优化方案指标比较图.png
        ├── 19_最终六区镜场与塔位平面图.png
        ├── 20_六区边界局部敏感性检验.csv
        ├── 21_六区边界局部敏感性图.png
        └── result3.xlsx
```

目录之间的关系如下：

- `task/` 是原始输入，不保存程序生成结果；
- `src/heliostat/` 是正式实现，三问入口只负责解析参数并调用对应模块；
- `docs/questions/` 解释模型、公式、数值方法和结果；
- `tests/` 检查共用物理模型、三问计算逻辑及 Excel 交付格式；
- `outputs/q1/`、`outputs/q2/`、`outputs/q3/` 保存完整计算产生的结果；
- 各输出目录中的 `01_第X问完整代码.py` 是便于独立运行和展示的合并程序，
  日常维护仍以 `src/` 下的模块化源码为准。

更详细的模块依赖和修改影响见
[`docs/WORK_BREAKDOWN.md`](docs/WORK_BREAKDOWN.md)，每个输出文件的内容见
[`outputs/README.md`](outputs/README.md)。

## 文档说明

三问文档按内容划分：

- `第一问.md`、`第二问.md`、`第三问.md` 说明问题目标、建模路线、计算流程、
  数值结果和结论边界；
- 三份“公式说明”集中列出符号、目标函数、物理公式、几何约束和验收公式；
- 三份 `q*-technical-notes.md` 记录代码结构、实现细节、搜索预算和复现约定；
- 第一问另有 `q1-plan.md` 和 `q1-validation.md`，分别保存简化实施规格和独立
  收敛验证。

## 环境安装

项目要求 Python 3.10 或更高版本。建议在仓库根目录创建独立虚拟环境。

macOS 或 Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主要依赖用途：

- NumPy：向量、镜场坐标和逐镜数组计算；
- SciPy：KDTree 邻镜搜索与 Sobol 准随机采样；
- OpenPyXL：读取题目附件并生成 `result2.xlsx`、`result3.xlsx`；
- Matplotlib：三问结果图；
- PySide6、PyVista、PyVistaQt：三维交互查看工具。

## 运行方式

以下命令在仓库根目录执行。macOS/Linux 可直接使用命令中的
`PYTHONPATH=src`；Windows PowerShell 可先执行 `$env:PYTHONPATH = "src"`，
再运行对应的 `python` 命令。

### 第一问

完整计算：

```bash
PYTHONPATH=src python src/solve_q1.py
```

快速检查：

```bash
PYTHONPATH=src python src/solve_q1.py \
  --months 6 \
  --times 12 \
  --limit-mirrors 20 \
  --shadow-grid 3 \
  --truncation-rays 8 \
  --output /tmp/q1-smoke
```

第一问完整结果写入 `outputs/q1/`，包含 60 个规定时刻的汇总、月平均、年平均、
1745 面镜子的单镜年平均、运行配置、验证表和两张结果图。

### 第二问

完整计算：

```bash
PYTHONPATH=src python src/solve_q2.py
```

快速检查：

```bash
PYTHONPATH=src python src/solve_q2.py \
  --smoke \
  --initial-samples 1 \
  --retained-starts 1 \
  --max-cycles 1 \
  --coarse-stride 20 \
  --extent-window 0 \
  --prune-rounds 0 \
  --output /tmp/q2-smoke
```

第二问完整结果写入 `outputs/q2/`。`result2.xlsx` 是按题目模板生成的提交文件，
其余文件记录双布局比较、最终镜位、月平均与年平均结果、单镜结果、加密验证、
四张结果图和绘图数据。

### 第三问

完整计算：

```bash
PYTHONPATH=src python src/solve_q3.py
```

快速检查：

```bash
PYTHONPATH=src python src/solve_q3.py \
  --smoke \
  --max-sweeps 1 \
  --output /tmp/q3-smoke
```

第三问完整结果写入 `outputs/q3/`。主要文件包括：

- `09_最终六区参数.csv`：最终塔位、Campo 参数与六区规格；
- `10_最终逐镜参数与坐标.csv`：1471 面镜子的最终坐标和规格；
- `11_正式结果比较.json`：原六区、收口前后与最终方案比较；
- `12_加密验收比较.json`：80 m、100 m 邻镜半径的加密结果；
- `14_局部收口检查.csv`：塔位包围、细扫和最细邻域检查；
- `15_论文结果与验证表.md`：主要结果、规格和验证表；
- `20_六区边界局部敏感性检验.csv`：18 个边界候选的完整结果；
- `result3.xlsx`：按题目模板生成的第三问提交文件。

修改第三问模块后，使用以下命令同步单文件程序：

```bash
python tool/build_q3_bundle.py
```

## 测试与代码检查

运行全部测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行代码检查：

```bash
ruff check src tests tool
```

测试覆盖以下内容：

- 太阳位置对称性、反射方向和集热器命中关系；
- 单镜阴影遮挡、异构镜面面积和截断效率单调性；
- 第一问月年汇总和单镜结果一致性；
- 第二问两种布局的几何约束、外边界评价和 `result2.xlsx`；
- 第三问六区回归、塔位模式、收口、18 个边界候选和 `result3.xlsx`。

## 结果口径

- 年平均值是题目规定的 12 个月、每月 5 个时刻，共 60 个状态的等权平均；
- 第一问只评价给定镜场，不受第二、三问 $42\ \mathrm{MW}$ 年平均功率下限
  约束；
- 第二、三问中的 $42\ \mathrm{MW}$ 是年平均下限，不要求每个月都达到该值；
- 快速检查、中精度搜索值和临时目录结果不能替代正式或加密复算结果；
- 第三问结论是当前六区结构、搜索预算和数值精度下的可靠可行改进，不表示严格
  局部最优或全局最优。

## 三维查看工具

```bash
python tool/heliostat3DApp.py
```

该工具用于观察镜场几何、镜面姿态和代表性中心光路。它显示的是可视化演示
功率，不包含正式求解中的完整阴影遮挡与截断计算，三问定量结果以 `outputs/`
中的正式文件为准。
