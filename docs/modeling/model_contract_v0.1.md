# B题统一模型契约 v1.0-fast-track

> 状态：Stage 4 正式冻结
> 文件名为兼容既有链接保留；正文版本决定效力。

## 1. 正式基准

- 舰船：二维匀速直线，速度 \(v_s=15\times0.514=7.71\rm\,m/s\)，等效探测
  圆盘半径 \(r_s=80\rm\,m\)。
- 导弹：恒速 \(320\rm\,m/s\) 的瞬时纯追踪；一阶惯性纯追踪与 PN 不是正式
  基准。
- 烟幕：中心固定，半径按题面 \(0\to120\rm\,m\)、保持 18 s、5 s 线性衰减。
- UAV：速度模长固定 \(28\rm\,m/s\) 的连续分段直线；全时段相对运动舰船距离
  不超过 12 km。
- 探测：距离不超过 8 km 且目标位于 \(\pm15^\circ\) 视场；瞬时纯追踪基准中
  光轴与视线重合。

## 2. 时间与事件语义

时间原点为任务决策开始。每枚弹 \(j\) 必须满足

\[
t_j^c\ge0,\qquad
t_j^d=t_j^c+2,\qquad
t_j^e=t_j^d+3.5.
\]

2 s 响应期间 UAV 正常飞行；3.5 s 从实际释放时刻开始计算。弹体继承释放瞬间
UAV 速度，故

\[
\boldsymbol c_j
=\boldsymbol p_j^d+3.5\,\boldsymbol v_u(t_j^d).
\]

同机相邻实际释放时刻至少相隔 1 s。基准场景禁止负时间预指令。

## 3. 覆盖与三值认证

令舰船圆盘为 \(S(t)=B(\boldsymbol s(t),80)\)，活动烟幕为
\(C_j(t)=B(\boldsymbol c_j,R_j(t))\)。正式防御条件为

\[
\forall t\in\mathcal D_i,\qquad
S(t)\subseteq\bigcup_{j\in\mathcal J}C_j(t).
\]

solver 仅生成候选，独立 verifier 只输出：

- `certified_feasible`：连续全称条件由保守界闭合；
- `certified_infeasible`：存在精确时间/空间裸露见证；
- `unresolved`：达到声明容差仍不能闭合。

局部 optimizer 成功不得替代 verifier 状态。

## 4. 目标层级

Q1：严格全窗口覆盖 → 最大认证覆盖总时长 → 最小最大连续裸露 → 最大最小覆盖
裕度 → 最小航程。

Q2：严格全窗口覆盖 → 最大最长无空档连续覆盖 → 最大认证覆盖总时长 → 最小最大
连续裸露 → 最小总裸露 → 最大联合覆盖裕度 → 最少弹药 → 最短航程。

## 5. 产物隔离

- 正式：`results/q1_rebuild/`、`results/q2_rebuild/`；
- 失锁耦合：`experimental_counterfactual`、`formal_baseline=false`；
- synthetic toy：`experiments/toy_demos/`、`formal_result=false`；
- 旧 `results/q1/` 不覆盖，PR #5 数值不继承。
