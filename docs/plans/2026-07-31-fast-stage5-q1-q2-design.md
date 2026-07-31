# 截止前 Stage 4 与 Q1–Q2 快速实施设计

## 1. 目标与边界

本轮从 `origin/agent/full-modeling-workflow-rebuild@2bd8f0e` 建立
`agent/fast-stage5-q1-q2`，base 保持研究分支，不合并 `main`。完成 Stage 4
冻结、Q1 正式重建和可提交版 Q2，停在 Q2 子检查点；不开始 Q3/Q4，不写最终论文
结论，不继承 PR #5 的代码或数值结果。

正式数值只写入 `results/q1_rebuild/`、`results/q2_rebuild/`，图只写入对应
`figures/*_rebuild/`。旧 `results/q1/` 保持不变。失锁耦合仅做只读重实现后的
少量 `experimental_counterfactual`，不进入正式矩阵。

## 2. 模型架构

公共层继续复用已有舰船匀速运动、瞬时纯追踪积分、题面烟幕半径、圆盘并集空间
认证和事件根求解。新增的 rebuild 层只组合公共原语：

1. `foundation` 冻结假设、符号、事件语义、三值认证和产物隔离。
2. `q1_rebuild` 采用两变量事件参数化：起爆时刻 \(t^e\) 与烟幕中心对应的舰船
   轨迹时刻 \(t^z\)。由 \(t^d=t^e-3.5\)、\(t^c=t^d-2\) 反推命令与释放时刻，
   并要求 \(t^c\ge0\)。无人机从 \(t=0\) 起正常飞行，以固定 28 m/s 的一段或
   两段折线到达释放点；弹体继承释放瞬间速度，3.5 s 后在固定中心起爆。
3. `q1_rebuild` 的 solver 用 Multi-start SLSQP、Sobol+SLSQP、SHGO、
   DE+SLSQP 在相同评估预算/种子下生成候选；独立 verifier 重新计算连续探测集、
   覆盖时长、最大裸露、最小裕度和航程，按用户批准的词典序排序。
4. `q2_rebuild` 用 1–3 枚弹的事件序列扩展 Q1 warm start。每枚弹有独立
   command/drop/burst 时刻和固定起爆中心；同一无人机的折线路径连续，投弹间隔
   不小于 1 s，作战半径与载弹量显式验证。
5. Q2 联合 verifier 在每个时间区间中点把舰船圆盘按
   \(v_s\Delta t/2\) 膨胀，并把各烟幕半径按最大半径变化率乘
   \(\Delta t/2\) 收缩。膨胀目标被收缩烟幕并集覆盖即可认证整个区间；精确时间点
   存在裸露见证即可认证不可行；达到时间容差仍不能判定则返回 `unresolved`。

## 3. 数据流与状态

`scripts/run_q1_rebuild.py` 先在 4 个代表方向上执行真实变量算法基准，再用胜出方法
运行 4 方向 × 4 初距正式矩阵，写 JSON/CSV/PNG。`scripts/run_q2_rebuild.py`
读取 Q1 rebuild 结果作为 warm starts，生成、连续精修并验证 1–3 弹方案，输出
当前最好已知方案及上下界。solver 的原生成功字段与 verifier 的
`certified_feasible`、`certified_infeasible`、`unresolved` 永不合并。

`state/decision_log.json` 先记录 Gate B 人工批准的 13 项决定和 Stage 4
尚未开始；Stage 4 产物通过自检后自动记录
`stage_4_status=passed`、`gate_c_status=approved_fast_track`、
`stage_5_started=true`。本轮最终仍处于 Stage 5 的 Q2 子检查点。

## 4. 失锁反事实

只读审查 `LSY-Q1-improve-lost&gravity` 的状态机与测试后，在本分支实现最小独立
反事实接口：`TRACKED/LOST` 两态、一阶转弯惯性 \(\tau_T\)、失锁转弯衰减
\(\tau_L\)、连续可见重捕获确认 \(T_R\)。仅对少量代表场景报告是否失锁、是否
重捕获和最小舰弹距离。其结果标记
`experimental_counterfactual=true`、`formal_baseline=false`。

## 5. 错误处理与证据等级

- 负时间预指令、响应/起爆时序错误、路径不连续、作战半径超限直接拒绝候选。
- optimizer 失败只影响该方法的候选生成，不自动判为模型不可行。
- 连续 verifier 找到精确时间/空间见证时才给出 `certified_infeasible`。
- 保守包络闭合全部区间时才给出 `certified_feasible`。
- 数值容差内仍无法闭合时保留 `unresolved`，并进入覆盖上界而非从统计中删除。
- 所有正式产物记录 git SHA、随机种子、方法预算、容差和运行时间。

## 6. 测试与验收

新增测试先失败、后实现：

1. Gate B 13 项决定、Stage 4/Gate C 状态与隔离边界；
2. 4×4 瞬时追踪正式矩阵和旧 144 场景兼容性；
3. Q1 固定响应语义、负时间拒绝、solver/verifier 分离、四方法统一预算；
4. Q2 多烟幕空间互补、连续时间联合认证、unresolved 保留、连续路径与投弹间隔；
5. 失锁反事实标签与参数敏感性输出；
6. 结果 schema、图表存在性和可复现性。

最终运行全量 pytest、Ruff、schema、场景、toy artifact、结果重放和
`git diff --check`，再更新 Draft PR。
