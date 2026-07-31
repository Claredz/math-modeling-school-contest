# Q1 连续验证与证据边界

验证器对每个候选重新计算：command/drop/burst 事件语义、路径连续性、固定 12 km
作战半径、释放瞬间速度和烟幕固定中心。连续覆盖验证按事件分段并使用 Lipschitz
包络；只能返回 `certified_feasible`、`certified_infeasible` 或 `unresolved`。

本轮正式几何矩阵为 4 个来袭方向 × 4 个初始距离，共 16 场景，结果在
`results/q1_rebuild/`，旧 `results/q1/` 不覆盖。一阶惯性、PN、有限漂移和失锁
耦合均为反事实或鲁棒扩展，不能把其数值写成题面事实。当前结果是给定 16 场景和
声明预算下的最好已知候选；不是全局最优证明。

正式 16 个场景仍全部为 `certified_infeasible`；修复没有改变正式 Q1 最佳候选的
事件时刻、覆盖时长、最大连续裸露、裕度或航程。失锁图只读取正式场景行中的同一
`candidate_id`，保持 command/drop/burst/center、释放位置、UAV path 和 smoke plan
不变，只改变失锁/重捕获模型；反事实带有 `fixed_decision=true`、
`changed_model=true`、`formal_baseline=false`，不进入 Q1 排名。

论文素材包括算法比较表 `q1_algorithm_benchmark.csv`、舰船/导弹/UAV 轨迹图、连续覆盖时间图、覆盖裕度曲线，以及固定正式候选的失锁反事实对照图。
