# 定日镜场优化设计

本项目对应 2023 年全国大学生数学建模竞赛 A 题，包含三问的正式源码、说明
文档、测试和可交付结果。当前仓库只保留每一问的最终路线。

## 当前正式结果

| 问题 | 镜子数 | 年平均功率 / MW | 单位面积输出 / (kW/m²) | 状态 |
| --- | ---: | ---: | ---: | --- |
| 第一问 | 1745 | 见 `outputs/q1/` | 见 `outputs/q1/` | 固定镜场评价完成 |
| 第二问 | 1469 | 42.044238 | 0.681068 | Campo 方案完成 |
| 第三问 | 1471 | 42.086215 | 0.693126 | 六区微调、正式收口及加密验收完成 |

第三问最终方案在 80 m 和 100 m 加密精度下均得到
$42.076085938\ \mathrm{MW}$、$0.692959144\ \mathrm{kW/m^2}$，稳定优于同口径
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

项目依赖记录在 `requirements.txt`。通用安装命令为：

```bash
python -m pip install -r requirements.txt
```

运行全部测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

本机若使用 Conda 环境 `agent`，可写成
`conda run -n agent env PYTHONPATH=src python ...`。环境名称不是项目要求。

## 第一问

第一问在题目给定的 1745 面镜场上计算 12 个月、每天 5 个规定时刻的余弦、
阴影遮挡、大气透射和截断效率，并汇总月平均、年平均与单镜年平均结果。

正式运行：

```bash
PYTHONPATH=src python src/solve_q1.py
```

快速 smoke：

```bash
PYTHONPATH=src python src/solve_q1.py \
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
PYTHONPATH=src python src/solve_q2.py
```

smoke：

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

正式结果位于 `outputs/q2/`，包含完整代码、双布局比较、最终坐标、月/年与单镜
结果、摘要、论文表格、加密验证、题目原名 `result2.xlsx`、四张图片和绘图数据。

## 第三问

第三问以原 1471 面六区阶梯方案为严格可行初值，依次执行：

1. 零增量正式回归；
2. 塔位模式 A/B 独立扫描；
3. Campo 参数 $D_1$、$g$ 扫描；
4. 六区 18 个规格变量正负敏感性；
5. 正式活跃变量筛选；
6. 最多两轮分块变步长回扫；
7. 塔位向北包围扫描、0.1 m 细扫及六变量一次最细邻域检查；
8. 原六区与候选的正式及 80/100 m 加密验收。

正式运行：

```bash
PYTHONPATH=src python src/solve_q3.py
```

smoke：

```bash
PYTHONPATH=src python src/solve_q3.py \
  --smoke \
  --max-sweeps 1 \
  --output /tmp/q3-smoke
```

第三问最终参数变化为：

- 塔位采用模式 B，$y_T$ 从 $-181.800054\ \mathrm{m}$ 调为
  $-176.300054\ \mathrm{m}$，向北移动 $5.5\ \mathrm{m}$；
- $D_1$ 减少 $0.1\ \mathrm{m}$；
- $g$ 增加 $0.01\ \mathrm{m/ring}$；
- G1 镜宽累计减少 $0.12\ \mathrm{m}$；
- G1 镜高减少 $0.02\ \mathrm{m}$；
- G2 安装高度累计降低 $0.12\ \mathrm{m}$；
- 其他六区规格保持原值。

最终仍为 1471 面镜子，总面积为 $60719.432482\ \mathrm{m^2}$，正式功率为
$42.086214740\ \mathrm{MW}$，单位面积输出为
$0.693125957\ \mathrm{kW/m^2}$。原搜索使用 89/150 个中精度候选和 12/12 个
正式筛选候选，另用 24 个正式候选完成塔位包围及一次最细邻域检查。本文只称
其为当前六区结构和预算下的可靠可行解，不声称严格收敛或全局最优。

正式结果位于 `outputs/q3/`，按 01--19 保存完整代码、扫描与敏感性数据、搜索及
收口轨迹、正式和加密比较、六区与逐镜参数、几何验证、论文表格和四张图片；
正式提交文件使用题目原名 `outputs/q3/result3.xlsx`。

生成第三问单文件展示稿：

```bash
python tool/build_q3_bundle.py
```

写论文同学优先查看 `docs/questions/第三问.md`、
`docs/questions/第三问公式说明.md`、`outputs/q3/15_论文结果与验证表.md`、
`outputs/q3/18_六组与优化方案指标比较图.png` 和
`outputs/q3/19_最终六区镜场与塔位平面图.png`；其余扫描与轨迹文件用于代码
复核。

详细方案见 [第三问.md](docs/questions/第三问.md)，公式见
[第三问公式说明.md](docs/questions/第三问公式说明.md)，实现约定见
[q3-technical-notes.md](docs/questions/q3-technical-notes.md)。

## 三维工具

```bash
conda run -n agent python tool/heliostat3DApp.py
```

该交互工具仅用于观察镜场，不作为论文正式功率来源；正式数值以
`outputs/q1/`、`outputs/q2/`、`outputs/q3/` 为准。
