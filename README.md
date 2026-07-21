# 定日镜场优化设计

本项目对应 2023 年全国大学生数学建模竞赛 A 题，包含三问的正式源码、说明
文档、测试和可交付结果。当前仓库只保留每一问的最终路线。

## 当前正式结果

| 问题 | 镜子数 | 年平均功率 / MW | 单位面积输出 / (kW/m²) | 状态 |
| --- | ---: | ---: | ---: | --- |
| 第一问 | 1745 | 见 `outputs/q1/` | 见 `outputs/q1/` | 固定镜场评价完成 |
| 第二问 | 1469 | 42.044238 | 0.681068 | Campo 方案完成 |
| 第三问 | 1471 | 42.088548 | 0.692971 | 六区微调及加密验收完成 |

第三问最终方案在 80 m 和 100 m 加密精度下均得到
$42.074138\ \mathrm{MW}$、$0.692733988\ \mathrm{kW/m^2}$，稳定优于同口径
原六区方案。

## 目录

```text
.
├── task/                         # 题面、附件和 result2/result3 模板
├── docs/
│   ├── WORK_BREAKDOWN.md
│   └── questions/
│       ├── 第一问.md
│       ├── 第一问公式说明.md
│       ├── 第二问.md
│       ├── 第二问公式说明.md
│       ├── 第三问.md
│       ├── 第三问公式说明.md
│       ├── q1-technical-notes.md
│       ├── q1-validation.md
│       ├── q2-technical-notes.md
│       └── q3-technical-notes.md
├── src/
│   ├── heliostat/
│   │   ├── config.py             # 共用物理与数值配置
│   │   ├── solar.py              # 太阳位置和 DNI
│   │   ├── geometry.py           # 镜面姿态与异构几何
│   │   ├── shadow.py             # 阴影遮挡
│   │   ├── truncation.py         # 截断效率
│   │   ├── q1/                   # 第一问评价与导出
│   │   ├── q2/                   # 第二问布局、搜索与导出
│   │   └── q3/                   # 第三问六区敏感性与局部优化
│   ├── solve_q1.py
│   ├── solve_q2.py
│   └── solve_q3.py
├── tests/
│   ├── test_core.py
│   ├── test_q1.py
│   ├── test_q2.py
│   └── test_q3.py
├── tool/
│   ├── build_q3_bundle.py
│   └── heliostat3DApp.py
└── outputs/
    ├── q1/
    ├── q2/
    └── q3/
```

## Python 环境

项目依赖记录在 `requirements.txt`。本仓库默认使用 Conda 环境 `agent`：

```bash
conda run -n agent python -m pip install -r requirements.txt
```

运行全部测试：

```bash
conda run -n agent env PYTHONPATH=src \
python -m unittest discover -s tests -v
```

## 第一问

第一问在题目给定的 1745 面镜场上计算 12 个月、每天 5 个规定时刻的余弦、
阴影遮挡、大气透射和截断效率，并汇总月平均、年平均与单镜年平均结果。

正式运行：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q1-mpl PYTHONPATH=src \
python src/solve_q1.py
```

快速 smoke：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q1-mpl PYTHONPATH=src \
python src/solve_q1.py \
  --months 6 \
  --times 12 \
  --limit-mirrors 20 \
  --shadow-grid 3 \
  --truncation-rays 8 \
  --output /tmp/q1-smoke
```

正式结果位于 `outputs/q1/`，包含完整代码、逐时刻/月/年结果、单镜结果、运行
配置、论文表格和两张正式图片。

## 第二问

第二问独立优化分区交错同心圆和改进 Campo 两种布局，统一使用 1 cm 镜间距
安全余量。最终采用 1469 面改进 Campo 镜场，正式年平均功率为
$42.044238\ \mathrm{MW}$，单位面积输出为
$0.681068\ \mathrm{kW/m^2}$。

正式运行：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q2-mpl PYTHONPATH=src \
python src/solve_q2.py
```

smoke：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q2-mpl PYTHONPATH=src \
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

正式结果位于 `outputs/q2/`，包含完整代码、双布局比较、最终坐标、月/年与单镜
结果、摘要、论文表格、加密验证、提交 Excel、四张图片和绘图数据。

## 第三问

第三问以原 1471 面六区阶梯方案为严格可行初值，依次执行：

1. 零增量正式回归；
2. 塔位模式 A/B 独立扫描；
3. Campo 参数 $D_1$、$g$ 扫描；
4. 六区 18 个规格变量正负敏感性；
5. 正式活跃变量筛选；
6. 最多两轮分块变步长回扫；
7. 原六区与候选的正式及 80/100 m 加密验收。

正式运行：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q3-mpl PYTHONPATH=src \
python src/solve_q3.py
```

smoke：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q3-mpl PYTHONPATH=src \
python src/solve_q3.py \
  --smoke \
  --max-sweeps 1 \
  --output /tmp/q3-smoke
```

第三问最终参数变化为：

- 塔位采用模式 B，$y_T$ 从 $-181.800054\ \mathrm{m}$ 调为
  $-178.800054\ \mathrm{m}$，向北移动 $3\ \mathrm{m}$；
- $D_1$ 减少 $0.1\ \mathrm{m}$；
- $g$ 增加 $0.01\ \mathrm{m/ring}$；
- G1 镜宽减少 $0.1\ \mathrm{m}$；
- G2 安装高度降低 $0.1\ \mathrm{m}$；
- 其他六区规格保持原值。

最终仍为 1471 面镜子，总面积为 $60736.356180\ \mathrm{m^2}$，正式功率为
$42.088548451\ \mathrm{MW}$，单位面积输出为
$0.692971247\ \mathrm{kW/m^2}$。中精度候选使用 89/150，正式候选使用
12/12。

正式结果位于 `outputs/q3/`，按 01--18 编号保存完整代码、扫描与敏感性数据、
搜索轨迹、正式和加密比较、六区与逐镜参数、几何验证、提交 Excel、论文表格和
三张正式图片。

生成第三问单文件展示稿：

```bash
conda run -n agent python tool/build_q3_bundle.py
```

详细方案见 [第三问.md](docs/questions/第三问.md)，公式见
[第三问公式说明.md](docs/questions/第三问公式说明.md)，实现约定见
[q3-technical-notes.md](docs/questions/q3-technical-notes.md)。

## 三维工具

```bash
conda run -n agent python tool/heliostat3DApp.py
```

该交互工具仅用于观察镜场，不作为论文正式功率来源；正式数值以
`outputs/q1/`、`outputs/q2/`、`outputs/q3/` 为准。
