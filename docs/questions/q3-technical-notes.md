# 第三问技术说明

本文档只供负责代码、运行和结果复核的人使用。论文方案与解释见
[`第三问.md`](第三问.md)，公式见 [`第三问公式说明.md`](第三问公式说明.md)。

## 1. 唯一代码结构

```text
src/
├── solve_q3.py
└── heliostat/q3/
    ├── __init__.py
    ├── _baseline.py
    ├── _optics.py
    ├── _workbook.py
    ├── six_group_baseline.json
    ├── model.py
    ├── tower_modes.py
    ├── evaluate.py
    ├── sensitivity.py
    ├── search.py
    ├── closure.py
    ├── export.py
    ├── plot.py
    └── solve.py
```

- `_baseline.py`：重建 1471 面 Campo 母场并执行异构几何检查；
- `_optics.py`：复用问题一光学模型，执行逐镜规格评价和缓存；
- `_workbook.py`：严格按题目模板写出 `result3.xlsx`；
- `model.py`、`tower_modes.py`：设计对象和塔位模式 A/B；
- `sensitivity.py`、`search.py`：规格筛选和两轮分块变步长搜索；
- `closure.py`：塔位包围、0.1 m 细扫和一次最细邻域检查；
- `export.py`、`plot.py`：结构化结果、提交表和论文图表；
- `solve.py`：唯一正式流程。

## 2. 输入路径

- `outputs/q2/07_最终方案摘要.json`：问题二 Campo 参数；
- `src/heliostat/q3/six_group_baseline.json`：原六区严格初值摘要；
- `task/A/result3.xlsx`：题目给定的第三问提交模板。

基线摘要只保存回归所需的塔位、六区规格、镜子数、面积和正式指标，不依赖旧
输出目录。

## 3. 运行命令

通用命令假定依赖已安装，并从仓库根目录执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_q3.py' -v
PYTHONPATH=src python src/solve_q3.py --smoke --max-sweeps 1 --output /tmp/q3-smoke
PYTHONPATH=src python src/solve_q3.py
python tool/build_q3_bundle.py
```

本机使用 Conda 环境 `agent` 时，可在命令前加：

```bash
conda run -n agent env PYTHONPATH=src
```

需要指定 Matplotlib 缓存目录时再增加
`MPLCONFIGDIR=/tmp/q3-mpl`。正式运行只写 `outputs/q3/`；smoke 必须写临时目录。

## 4. 回归门槛

搜索开始前必须复现：

```text
mirror_count = 1471
total_area_m2 = 60777.39103369038
annual_power_mw = 42.051608025429616
unit_area_power_kw_m2 = 0.6918955767963023
```

逐镜坐标、宽度、高度和安装高度最大绝对误差为 0；面积允许 $10^{-6}$ 量级
浮点误差，功率允许 $10^{-6}\ \mathrm{MW}$，$q$ 允许
$10^{-9}\ \mathrm{kW/m^2}$。

## 5. 候选预算与停止条件

- 塔位初扫：模式 A/B 各 7 个中精度点，各取 2 个正式候选；
- Campo：$D_1$、$g$ 各 5 个一维点，加 $3\times3$ 组合；
- 规格：18 个变量各正负扰动，共 36 个中精度候选；
- 正式规格方向：6 个不同变量；
- 原搜索上限：中精度 150、正式筛选 12、联合回扫 2 轮；
- 本次原搜索实际使用：中精度 89、正式筛选 12；
- 正式收口：塔位找到首个北侧下降点后做 0.1 m 细扫，再完成一次六变量最细
  正负检查；本次使用 24 个正式候选。

塔位必须找到下降点，否则正式运行硬失败。六变量最细检查按附件要求只执行一次；
即使接受移动，也不自动扩张成新的长期优化。

## 6. 输出结构

唯一正式输出目录为 `outputs/q3/`：

| 文件 | 用途 |
| --- | --- |
| `01_第三问完整代码.py` | 可独立运行的单文件展示稿 |
| `02_六组回归结果.json` | 初值回归和预算 |
| `03_塔位两种语义扫描.csv` | 模式 A/B 初扫 |
| `04_Campo几何粗扫.csv` | $D_1$、$g$ 扫描 |
| `05_规格参数敏感性.csv` | 18 个规格变量正负扰动 |
| `06_活跃变量集合.json` | 活跃变量、预算和收口状态 |
| `07_局部搜索轨迹.csv` | 两轮分块搜索接受轨迹 |
| `08_正式候选比较.csv` | 正式候选统一比较 |
| `09_最终六区参数.csv` | 最终六区规格 |
| `10_最终逐镜参数与坐标.csv` | 1471 面镜子最终数据 |
| `11_正式结果比较.json` | 基线、收口前、收口后和最终结果 |
| `12_加密验收比较.json` | 80/100 m 加密结果 |
| `13_几何约束验证.json` | 最终几何合法性 |
| `14_局部收口检查.csv` | 塔位包围、细扫和六变量最细检查 |
| `15_论文结果与验证表.md` | 论文表格 |
| `16_参数敏感性图.png` | 各阶段对应基准的敏感性图 |
| `17_六区宽高与安装高度图.png` | 六区规格图 |
| `18_六组与优化方案指标比较图.png` | 正式及加密比较图 |
| `19_最终六区镜场与塔位平面图.png` | 正文镜场与塔位图 |
| `result3.xlsx` | 题目要求原名的正式提交文件 |

`result3.xlsx` 必须保持模板工作表、表头和八列顺序，共 1472 行，其中数据区
1471 行，镜号连续为 1--1471。

## 7. 调试约定

- 正式数值只能来自完整 12 个月、每天 5 个规定时刻的正式或加密精度；
- smoke 数值不能写入论文或正式输出；
- 中精度只用于排序和局部接受，最终选择必须经过正式及加密验收；
- 敏感性图中塔位、Campo、规格分别相对各自阶段基准，禁止混用基准；
- 修改正式源码后必须重新执行 `tool/build_q3_bundle.py`；
- 提交前运行全套测试、Ruff、单文件编译和 Markdown 数学格式检查。
