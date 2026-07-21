# 第一问建模实施方案

本文档用较短篇幅列出第一问的固定参数、数值方法、程序执行顺序和必要检查，
用于快速核对实现规格。完整建模过程见 [`第一问.md`](第一问.md)，公式推导见
[`第一问公式说明.md`](第一问公式说明.md)。

## 1. 问题目标

已知 1745 面定日镜的位置、尺寸和安装高度，计算题目规定的 60 个时刻下：

- 平均光学效率；
- 平均余弦效率；
- 平均阴影遮挡效率；
- 平均截断效率；
- 镜场输出热功率；
- 单位镜面面积输出热功率；
- 各项指标的月平均和年平均结果。

第一问只评价给定镜场，不需要优化定日镜的位置和尺寸。

---

## 2. 最终采用的方法

| 模块 | 方法 |
| --- | --- |
| 太阳位置与 DNI | 题目附录公式 |
| 镜面姿态 | 三维向量反射定律 |
| 余弦效率 | 向量点积 |
| 阴影遮挡效率 | $15\times15$ 规则网格射线追踪 |
| 候选遮挡镜 | 包围球方向筛选 |
| 大气透射率 | 题目经验公式 |
| 截断效率 | 256 条 Sobol 联合采样光线 |
| 镜场功率 | 所有定日镜功率求和 |
| 月均、年均 | 对规定时刻等权平均 |

总体流程为：

```text
读取镜面坐标
    ↓
计算太阳方向和 DNI
    ↓
计算所有镜面的姿态
    ↓
计算余弦效率
    ↓
计算阴影遮挡效率
    ↓
计算大气透射率
    ↓
计算截断效率
    ↓
计算单镜和镜场功率
    ↓
计算月平均与年平均
```

---

## 3. 固定参数

第 $i$ 面定日镜的地面坐标为 $(x_i,y_i)$，镜面中心为 $\boldsymbol c_i=(x_i,y_i,4.5).$ 定日镜尺寸为 $6.2\ \mathrm{m}\times6.2\ \mathrm{m},$ 单镜面积为 $A=6.2^2=38.44\ \mathrm{m^2}.$ 集热器中心为 $\boldsymbol C=(0,0,86).$ 集热器为半径 $4\ \mathrm{m}$、高度 $8\ \mathrm{m}$ 的圆柱侧面： $x^2+y^2=16,\qquad 82\le z\le90.$ 镜面反射率取 $\eta_{\mathrm{ref}}=0.92.$

---

## 4. 太阳位置和镜面姿态

每月 21 日计算以下五个时刻： $9{:}00,\quad10{:}30,\quad12{:}00,\quad13{:}30,\quad15{:}00.$ 全年共计算 $12\times5=60$ 个时刻。

每个时刻根据题目附录计算太阳方向单位向量 $\boldsymbol s$ 和 $DNI$。规定 $\boldsymbol s$ 为从定日镜指向太阳的方向。

第 $i$ 面定日镜指向集热器中心的单位向量为 $\boldsymbol r_i = \frac{\boldsymbol C-\boldsymbol c_i} {\left\|\boldsymbol C-\boldsymbol c_i\right\|}.$ 镜面法向为 $\boxed{ \boldsymbol n_i = \frac{\boldsymbol s+\boldsymbol r_i} {\left\|\boldsymbol s+\boldsymbol r_i\right\|} }$ 令竖直方向为 $\boldsymbol k=(0,0,1)$，建立镜面局部坐标系： $\boldsymbol u_i = \frac{\boldsymbol k\times\boldsymbol n_i} {\left\|\boldsymbol k\times\boldsymbol n_i\right\|},$ $\boldsymbol v_i=\boldsymbol n_i\times\boldsymbol u_i.$ 镜面上的任意一点为 $\boldsymbol q = \boldsymbol c_i+a\boldsymbol u_i+b\boldsymbol v_i, \qquad -3.1\le a,b\le3.1.$

---

## 5. 各项效率

### 5.1 余弦效率

余弦效率为 $\boxed{ \eta_{\cos,i} = \boldsymbol s\cdot\boldsymbol n_i }$。镜面理论截获的太阳功率为 $P_{\mathrm{capture},i} = DNI\cdot A\cdot\eta_{\cos,i}.$

### 5.2 阴影遮挡效率

将每面定日镜划分为 $15\times15$ 个等面积网格，共 225 个采样点。

对每个采样点分别检查：

1. 沿 $\boldsymbol s$ 方向发射射线，判断太阳光是否被其他定日镜挡住；
2. 沿 $\boldsymbol r_i$ 方向发射射线，判断反射光在到达集热器前是否被其他定日镜挡住。

若一个采样点的入射光和反射光都未被遮挡，则记为有效点。阴影遮挡效率为 $\boxed{ \eta_{sb,i} = \frac{\text{有效采样点数}}{225} }$ 阴影损失和反射遮挡损失必须在采样点层面取并集，不能直接把两个损失比例相加。

### 5.3 候选遮挡镜筛选

为避免每条射线检查其余全部 1744 面定日镜，先用镜面包围球进行方向筛选。

正方形定日镜的包围球半径取其半对角线： $R_b = \frac{\sqrt{6.2^2+6.2^2}}{2} \approx4.384\ \mathrm{m}.$ 只有位于射线前方，并且中心到射线的垂直距离小于约 $2R_b$ 的定日镜，才进入精确的射线—矩形求交检查。

判断反射遮挡时，还必须保证遮挡交点位于集热器之前。

### 5.4 大气透射率

第 $i$ 面定日镜到集热器中心的距离为 $d_{HR,i} = \left\|\boldsymbol C-\boldsymbol c_i\right\|.$ 大气透射率为 $\boxed{ \eta_{at,i} = 0.99321 -0.0001176d_{HR,i} +1.97\times10^{-8}d_{HR,i}^2 }$ 定日镜位置不变，因此该效率只需预计算一次。

### 5.5 截断效率

使用 256 条 Sobol 联合采样光线。每条样本同时确定：

- 镜面上的一个反射位置；
- 太阳圆盘内的一个入射方向。

太阳角半径取 $\theta_\odot=4.65\ \mathrm{mrad}.$ 对每条采样光线，根据反射定律计算反射方向，再判断它是否命中圆柱形集热器侧面。

若 256 条光线中有 $N_{\mathrm{hit}}$ 条命中，则 $\boxed{ \eta_{\mathrm{trunc},i} = \frac{N_{\mathrm{hit}}}{256} }$

---

## 6. 单镜和镜场功率

第 $i$ 面定日镜的总光学效率为 $\boxed{ \eta_i = \eta_{\cos,i} \eta_{sb,i} \eta_{at,i} \eta_{\mathrm{trunc},i} \eta_{\mathrm{ref}} }$ 单镜输出功率为 $\boxed{ P_i(t) = DNI(t)\cdot A\cdot\eta_i(t) }$ 镜场总输出功率为 $\boxed{ E_{\mathrm{field}}(t) = \sum_{i=1}^{1745}P_i(t) }$ 平均光学效率应先逐镜计算效率乘积，再求平均： $\overline{\eta}_{\mathrm{opt}}(t) = \frac{1}{1745} \sum_{i=1}^{1745}\eta_i(t).$ 不能直接把各项平均效率相乘。

单位镜面面积输出功率为 $q(t) = \frac{E_{\mathrm{field}}(t)} {1745\times38.44}.$

---

## 7. 月平均和年平均

第 $m$ 月五个时刻的平均输出功率为 $\overline E_m = \frac{1}{5} \sum_{k=1}^{5}E_{m,k}.$ 全年平均输出功率为 $\boxed{ \overline E_{\mathrm{year}} = \frac{1}{60} \sum_{m=1}^{12} \sum_{k=1}^{5}E_{m,k} }$ 其他效率指标也按照相同方式计算月平均和年平均。

必须先计算每个时刻的功率，再进行平均，不能使用“平均 DNI × 平均效率”代替年平均功率。

---

## 8. 程序执行顺序

```text
1. 读取1745面定日镜的坐标
2. 生成镜面中心三维坐标
3. 预计算集热器方向、传播距离和大气透射率
4. 遍历12个月和每天5个规定时刻
5. 计算该时刻的太阳方向和DNI
6. 一次性计算全部镜面的法向和局部坐标系
7. 用15×15规则网格计算阴影遮挡效率
8. 用256条Sobol光线计算截断效率
9. 计算每面定日镜的总效率和输出功率
10. 汇总镜场时刻指标
11. 计算月平均和年平均
12. 输出题目要求的结果表
```

---

## 9. 必要的正确性检查

程序至少完成以下检查：

1. **反射方向检查** $\left\| \operatorname{reflect}(-\boldsymbol s,\boldsymbol n_i) -\boldsymbol r_i \right\| <10^{-8}.$

2. **效率范围检查** $0\le\eta\le1.$

3. **单镜无邻居检查**

   只有一面定日镜时，应满足 $\eta_{sb}=1.$

4. **集热器半径单调性检查**

   集热器半径由 $3\ \mathrm{m}$ 增加到 $4\ \mathrm{m}$、$5\ \mathrm{m}$ 时，截断效率不应下降。

5. **采样收敛检查**

   用更高网格分辨率或更多 Sobol 光线抽查少量代表性定日镜，确认结果变化已经足够小。

---

## 10. 最终结论

第一问最终采用：

> 题目公式解析计算太阳位置和 DNI，利用向量反射定律确定镜面姿态，采用 $15\times15$ 规则网格计算阴影遮挡效率，采用 256 条 Sobol 联合采样光线计算截断效率，最后逐镜、逐时刻计算功率并汇总月平均和年平均结果。

完整能量模型为 $\boxed{ P_i(t) = DNI(t) \cdot A \cdot\eta_{\cos,i}(t) \cdot\eta_{sb,i}(t) \cdot\eta_{at,i} \cdot\eta_{\mathrm{trunc},i}(t) \cdot\eta_{\mathrm{ref}} }$
