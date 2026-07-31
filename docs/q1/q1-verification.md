# Q1 连续验证与证据边界

验证器对每个候选重新计算：command/drop/burst 事件语义、路径连续性、固定 12 km
作战半径、释放瞬间速度和烟幕固定中心。连续覆盖验证按事件分段并使用 Lipschitz
包络；只能返回 `certified_feasible`、`certified_infeasible` 或 `unresolved`。

本轮正式几何矩阵为 4 个来袭方向 × 4 个初始距离，共 16 场景，结果在
`results/q1_rebuild/`，旧 `results/q1/` 不覆盖。一阶惯性、PN、有限漂移和失锁
耦合均为反事实或鲁棒扩展，不能把其数值写成题面事实。当前结果是给定 16 场景和
声明预算下的最好已知候选；不是全局最优证明。
