# 第三问技术说明

六区阶梯参数微调实现约定

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
    ├── export.py
    ├── plot.py
    └── solve.py
```

三个下划线模块是当前方案仍然需要的内部基础设施：

- `_baseline.py`：从问题二参数重建 1471 面 Campo 母场并执行异构几何检查；
- `_optics.py`：复用问题一光学模型，按逐镜宽、高、安装高度统一评价；
- `_workbook.py`：按题目 `result3.xlsx` 模板写入最终逐镜数据。

其余文件对应当前正式流程，不保留其他第三问模型目录。

## 2. 输入

- `outputs/q2/07_最终方案摘要.json`：问题二 Campo 几何参数；
- `src/heliostat/q3/six_group_baseline.json`：原六区严格初值的最小只读摘要；
- `task/A/result3.xlsx`：第三问提交模板。

基线摘要只保存塔位、六区规格、镜子数、面积和正式年平均指标，不保存旧搜索
轨迹或旧输出包。

## 3. 运行

专项测试：

```bash
conda run -n agent env PYTHONPATH=src \
python -m unittest discover -s tests -p 'test_q3.py' -v
```

smoke：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q3-mpl PYTHONPATH=src \
python src/solve_q3.py \
  --smoke \
  --max-sweeps 1 \
  --output /tmp/q3-smoke
```

正式运行：

```bash
conda run -n agent env MPLCONFIGDIR=/tmp/q3-mpl PYTHONPATH=src \
python src/solve_q3.py
```

生成单文件展示稿：

```bash
conda run -n agent python tool/build_q3_bundle.py
```

## 4. 回归门槛

搜索开始前必须精确复现：

```text
mirror_count = 1471
total_area_m2 = 60777.39103369038
annual_power_mw = 42.051608025429616
unit_area_power_kw_m2 = 0.6918955767963023
```

逐镜坐标、宽度、高度和安装高度最大绝对误差必须为 0。

## 5. 候选预算

- 塔位：模式 A/B 各 7 个中精度点，各 2 个正式候选；
- Campo：$D_1$ 与 $g$ 各 5 个一维点，加 $3\times3$ 组合；
- 规格：18 个变量各正负扰动，共 36 个中精度候选；
- 正式规格方向：6 个不同变量；
- 最终局部候选：1 个正式验收；
- 中精度总上限：150；
- 正式候选总上限：12；
- 联合回扫上限：2 轮。

## 6. 输出

唯一正式输出目录为 `outputs/q3/`，编号 01--18：

1. 完整代码；
2. 六区基线回归；
3. 两种塔位语义扫描；
4. Campo 几何扫描；
5. 规格敏感性；
6. 活跃变量；
7. 局部搜索轨迹；
8. 正式候选比较；
9. 最终六区参数；
10. 最终逐镜参数与坐标；
11. 正式结果比较；
12. 80/100 m 加密比较；
13. 几何约束；
14. 提交 Excel；
15. 论文结果表；
16--18. 三张正式图片。
