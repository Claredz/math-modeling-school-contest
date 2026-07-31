# Q2 多弹事件参数化模型

## 范围

Q2 在 Stage 4 冻结的正式基线之上允许同一架 UAV 使用 1–3 枚弹。每枚弹的决策变量为指令、实际投弹和起爆时刻、起爆中心以及投弹次序；UAV 路径由连续的分段直线连接器生成。2 秒响应延时定义为指令到实际释放，3.5 秒延时从实际释放开始计时，基准场景禁止负时间指令。

烟幕中心在正式基线中固定于舰船轨迹的事件时刻；Q2 的多弹联合覆盖使用同一舰船 80 m 等效探测圆盘和题面半径衰减。候选生成允许错位起爆，但不改变固定中心和事件语义。

## 变量与约束

- 每枚弹：`command_time`, `drop_time`, `burst_time`，满足 `drop-command=2`、`burst-drop=3.5`。
- 同一 UAV 的投弹次序严格递增，并保留至少 1 s 的释放间隔。
- 每个投弹点由 UAV 速度和两段直线连接器产生；独立 verifier 检查 12 km 作战半径。
- 目标是检测窗口上的联合覆盖，而不是局部优化器的原生成功标志。

本轮 Q2 是可提交的有界候选流程，不宣称全局精确最优。结果目录 `results/q2_rebuild/` 与旧 `results/q1/` 隔离。

## 目标与约束编号

令 \(z_j\in\{0,1\}\) 表示第 \(j\) 枚弹是否使用，事件和中心组成
\(\mathbf{x}_2=(z_j,t_j^c,t_j^d,t_j^e,c_j,p_j^d,\pi)\)。目标词典序为
\(\mathcal{J}_2=(I_{\rm full},T_{\rm cov}^{\rm lower},-T_{\rm gap}^{\max},-T_{\rm exposed},G_{\rm joint},-\sum_jz_j,-L_{\rm UAV})\)。
约束为：

1. `(Q2-C1)` \(t_j^d=t_j^c+2,\;t_j^e=t_j^d+3.5,\;t_j^c\ge0\)；
2. `(Q2-C2)` \(1\le\sum_jz_j\le3\)，同一 UAV 的投弹间隔至少 1 s；
3. `(Q2-C3)` 每个 \(p_j^d\) 可由连续 UAV 路径到达，且作战半径不超过 12 km；
4. `(Q2-C4)` 起爆中心、路径速度和事件时刻一致；
5. `(Q2-C5)` 烟幕并集对全部检测时间和 80 m 圆盘执行联合 verifier；认证覆盖区间、认证裸露区间和 `unresolved` 区间分别保存，连续 separation 无法关闭时不伪造裸露时长。

结果字段中，`joint_coverage_lower_s` 是联合覆盖下界，`best_single_smoke_coverage_lower_s` 来自同一场景、同一 verifier 的最佳单弹候选，且

\[
G_{\rm joint}=T_{\rm cov,joint}^{\rm lower}-T_{\rm cov,single}^{\rm lower}.
\]

`maximum_continuous_exposure_s` 明确表示认证裸露下界；未决导致的可能更大连续裸露通过 `maximum_exposure_upper_s` 单独表达。
